import geopandas as gpd
import pandas as pd
import math

from rdflib import Graph, Namespace
from rdflib import Literal, URIRef, BNode
from rdflib import RDF
from rdflib.namespace import FOAF, TIME, XSD, RDFS


class RDFBuilder() :     

    ### METODI PROTETTI ###
    
    # *** Some functions used to simplify the insertion of enriched stop/moves in the KG... *** #

    # Given a graph g, an agent identifier uid, and a subtrajectory identifier tid, this method
    # returns the nodes in g representing the agent corresponding to uid, and the nodes associated
    # with the logical and raw trajectories corresponding to uid and tid. 
    def find_trajectories_from_graph(self, uid, tid) :

        user = self.g.value(predicate = FOAF.name, object = Literal(uid), any = False)
        set_traj = self.g.subjects(predicate = self.STEP.hasID, object = Literal(tid))
        traj = None
        for t in set_traj :
            if (user, self.STEP.hasTrajectory, t) in self.g : 
                traj = t
                break
        raw_traj = self.g.value(subject = traj, predicate = self.STEP.hasRawTrajectory, any = False)
        # print(f"Raw => {raw_traj}")

        return user, traj, raw_traj
    
    
    # Given a graph g, an agent identifier uid, this method returns the node in g representing the agent corresponding to uid. 
    def find_user_from_graph(self, uid) :

        return self.g.value(predicate = FOAF.name, object = Literal(uid), any = False)

    
    # Given a graph g, a node raw_traj representing a specific raw subtrajectory, and a start and end time instants,
    # find the corresponding instant and location nodes in the raw trajectory. 
    def find_instants_locations_start_end(self, raw_traj, start_t, end_t) :
    
        start = Literal(start_t)
        end = Literal(end_t)
        
        instant_start = None
        instant_end = None
        location_start = None
        location_end = None
        set_fixes = self.g.objects(subject = raw_traj, predicate = self.STEP.hasFix)
        for f in set_fixes :

            instant = self.g.value(subject = f, predicate = self.STEP.atTime, any = False)
            time = self.g.value(subject = instant, predicate = TIME.inXSDDateTime, any = False)

            if time == start : 
                #print("Trovato start!")
                instant_start = instant
                location_start = self.g.value(subject = f, predicate = self.STEP.hasLocation, any = False)

            if time == end : 
                #print("Trovato end!")
                instant_end = instant
                location_end = self.g.value(subject = f, predicate = self.STEP.hasLocation, any = False)

            if (instant_start != None) & (instant_end != None) : break


        return instant_start, location_start, instant_end, location_end
    
    
    
    ### COSTRUTTORE CLASSE ###
    
    def __init__(self) :
    
        """Costruttore della classe.
        """
        
        # Impostazione campi base.
        self.g = Graph()
    
    
        # Pull in the ontologies we will use while populating the knowledge graph.
        self.STEP = Namespace('http://purl.org/net/step_specialized#')
        self.GEO = Namespace('http://www.w3.org/2003/01/geo/wgs84_pos#')


        # Bind the above references to the ontologies to the graph.
        self.g.bind('step_specialized', self.STEP)
        self.g.bind('geo', self.GEO)
        
        
        # Define a dictionary containing the mapping to be used for transport mode.
        self.dic_sys_stop = {'home' : self.STEP.Home,
                             'work' : self.STEP.Work,
                             'other' : self.STEP.SystematicStop}
        
        
        # Define a dictionary containing the mapping to be used for transport mode.
        self.dic_moves = {0 : self.STEP.Walk,
                          1 : self.STEP.Bike,
                          2 : self.STEP.Bus,
                          3 : self.STEP.Car,
                          4 : self.STEP.Subway,
                          5 : self.STEP.Train,
                          6 : self.STEP.Taxi}

    
    
    ### METODI PUBBLICI ###
    
    def add_trajectories(self, df_raw_traj) :
        
        df_raw_traj['datetime'] = pd.to_datetime(df_raw_traj['datetime'], utc = True)
        # print(df_raw_traj)
        # print(df_raw_traj.info())


        # Retrieve the list of users from the dataframe.
        list_users_cleaned = df_raw_traj['uid'].unique()
        # print(f"List of unique user IDs: {list_users_cleaned}")
        print(f"Number of users whose trajectories will be added to the RDF graph: {df_raw_traj['uid'].nunique()}")
        print(f"Number of trajectories that will be added to the RDF graph: {df_raw_traj['tid'].nunique()}")
        
        
        # *** Build the part of the KG related to the users and their trajectories. *** #
        for t in list_users_cleaned :
        
            view = df_raw_traj.loc[df_raw_traj['uid'] == t]
            #display(view)
            
            # Create the agent node.
            agent = URIRef('http://example.org/user_' + str(t) + '/')
            # print(f"Considering user {t} (graph node with ID {agent})")
            
            self.g.add((agent, RDF.type, FOAF.Agent))
            self.g.add((agent, FOAF.name, Literal(t)))
            
            # Associate to the agent all its trajectories
            list_sub_t = view["tid"].unique()
            # print(f"List of subtrajectory IDs associated with the user {t}: {list_sub_t}")
            #print(f"Number of trajectories associated with the user {t}: {len(list_sub_t)}")
            #print(f"Number of samples associated with the user {t}: {view.shape[0]}")
            
            for s in list_sub_t :
                
                # Create the "logical" trajectory which will hold the raw trajectory + its semantic aspects.
                URI_traj = 'http://example.org/user_' + str(t) + '/traj_' + str(s) + '/'
                traj = URIRef(URI_traj)
                self.g.add((agent, self.STEP.hasTrajectory , traj))
                self.g.add((traj, RDF.type, self.STEP.Trajectory))
                self.g.add((traj, self.STEP.hasID, Literal(s))) # Save the ID of this trajectory.
                
                
                # Create the raw trajectory
                seg = URIRef(URI_traj + 'raw/')
                self.g.add((seg, RDF.type, self.STEP.RawTrajectory))
                self.g.add((traj, self.STEP.hasRawTrajectory, seg))
                
                
                # Extract the info associated with the fixes.
                view_2 = view.loc[view['tid'] == s]
                list_lat = view_2['lat']
                list_long = view_2['lng']
                list_time = view_2['datetime']
                list_fix = zip(list_lat, list_long, list_time)
                # print(f"Generating trajectory {s} for user {t} (samples: {len(view_2)})")
                
                
                cnt = 0
                for (lat, long, time) in list_fix :
                    fix = URIRef(URI_traj + 'raw/fix_' + str(cnt) + '/')
                    point = URIRef(URI_traj + 'raw/pt_' + str(cnt) + '/')
                    instant = URIRef(URI_traj + 'raw/inst_' + str(cnt) + '/')
                    
                    self.g.add((fix, RDF.type, self.STEP.Fix))
                    self.g.add((seg, self.STEP.hasFix, fix)) # Associate fix to raw trajectory segment.
                    
                    self.g.add((point, RDF.type, self.STEP.Point))
                    self.g.add((fix, self.STEP.hasLocation, point)) # Associate point to fix.
                    
                    # Associate coordinates to the point.
                    self.g.add((point, self.GEO.lat, Literal(lat)))
                    self.g.add((point, self.GEO.long, Literal(long)))
                    
                    # Associate time instant to the fix.
                    self.g.add((instant, RDF.type, self.STEP.Instant))
                    self.g.add((fix, self.STEP.atTime, instant))
                    self.g.add((instant, TIME.inXSDDateTime, Literal(time)))
                    
                    cnt = cnt + 1
                    
                    
    def add_occasional_stops(self, df_occasional_stops) :
        
        df_occasional_stops['datetime'] = pd.to_datetime(df_occasional_stops['datetime'], utc = True)
        df_occasional_stops['leaving_datetime'] = pd.to_datetime(df_occasional_stops['leaving_datetime'], utc = True)
        # print(f"Dataframe of the occasional stops used to generate RDF triples: {df_occasional_stops.info()}")

        view_stop_data = df_occasional_stops[['stop_id', 'uid', 'tid', 'datetime', 'leaving_datetime', 'osmid', 'element_type', 'name', 'wikidata', 'category', 'distance']]
        #print(view_stop_data)
        print(f"Number of occasional stops: {view_stop_data['stop_id'].nunique()}")
        
        gb = view_stop_data.groupby(['stop_id'])
        for key in gb.groups.keys() :
            
            cnt = 0
            
            # Retrieve the rows associated with this stop.
            group = gb.get_group(key)

            # Retrieve the main properties of this stop.
            uid = group['uid'].iloc[0]
            tid = group['tid'].iloc[0]
            start = group['datetime'].iloc[0]
            end = group['leaving_datetime'].iloc[0]
            list_POI = group[['osmid', 'element_type', 'name', 'wikidata', 'category', 'distance']]

            # print(f"{key} -- {uid} -- {tid} -- {start} -- {end}")
            # print(list_POI)
            # print(list_POI.info())
            # break

            # Find the nodes in the graph associated with the "uid" and "tid" identifiers.
            user, traj, raw_traj = self.find_trajectories_from_graph(uid, tid)

            # Find the nodes in the graph corresponding to the instants and locations associated with this move.
            instant_start, location_start, instant_end, location_end = \
                self.find_instants_locations_start_end(raw_traj, start, end)


            # *** Now, create all the triples neeed to semantically enrich the trajectory with this move. *** #
            # Feature node.
            URI_feat = 'http://example.org/user_' + str(uid) + '/traj_' + str(tid) + '/feature_occ_stop/'
            feature = URIRef(URI_feat)
            self.g.add((feature, RDF.type, self.STEP.Feature))
            self.g.add((traj, self.STEP.hasFeature, feature))

            # Episode node.
            URI_episode = URI_feat + str(key) + '/'
            episode = URIRef(URI_episode)
            self.g.add((episode, RDF.type, self.STEP.Episode))
            self.g.add((feature, self.STEP.hasEpisode, episode))

            # Semantic description (Occasional Stop) node.
            stop_desc = URIRef(URI_episode + 'desc/')
            self.g.add((stop_desc, RDF.type, self.STEP.OccasionalStop))
            self.g.add((stop_desc, RDFS.subClassOf, self.STEP.Stop))
            self.g.add((episode, self.STEP.hasSemanticDescription, stop_desc))

            # Link this Occasional Stop with all the POIs that may be associated with it.
            for osm, type, name, wd, cat, distance in zip(list_POI['osmid'], list_POI['element_type'], list_POI['name'], list_POI['wikidata'], list_POI['category'], list_POI['distance']) :
                if not pd.isna(osm) :
                    poi = URIRef('http://example.org/poi_' + str(osm) + '/')
                    self.g.add((poi, RDF.type, self.STEP.PointOfInterest))
                    self.g.add((poi, self.STEP.hasOSMValue, Literal(str(osm))))
                    self.g.add((poi, self.STEP.hasOSMName, Literal(str(name))))
                    self.g.add((poi, self.STEP.hasOSMType, Literal(str(type))))
                    self.g.add((poi, self.STEP.hasOSMCategory, Literal(str(cat))))
                    if not pd.isna(wd): self.g.add((poi, self.STEP.hasWDValue, URIRef("http://www.wikidata.org/entity/" + str(wd))))
                    self.g.add((stop_desc, self.STEP.hasPOI, poi))

            # Spatiotemporal extent.
            st_extent = URIRef(URI_episode + 'extent/')
            self.g.add((st_extent, RDF.type, self.STEP.SpatiotemporalExtent))
            self.g.add((episode, self.STEP.hasExtent, st_extent))

            # Starting keypoint to associate to the spatiotemporal extent.
            start_kp = URIRef(URI_episode + 'extent/skp/')
            self.g.add((start_kp, RDF.type, self.STEP.KeyPoint))
            self.g.add((start_kp, self.STEP.atTime, instant_start))
            self.g.add((start_kp, self.STEP.hasLocation, location_start))
            self.g.add((st_extent, self.STEP.hasStartingPoint, start_kp))

            # Ending keypoint to associate to the spatiotemporal extent.
            end_kp = URIRef(URI_episode + 'extent/ekp/')
            self.g.add((end_kp, RDF.type, self.STEP.KeyPoint))
            self.g.add((end_kp, self.STEP.atTime, instant_end))
            self.g.add((end_kp, self.STEP.hasLocation, location_end))
            self.g.add((st_extent, self.STEP.hasEndingPoint, end_kp))

            
            
    def add_systematic_stops(self, df_sys_stops) :
        
        # print(df_sys_stops)
        # print(df_sys_stops.info())
        
        
        print(f"Number of systematic stops: {df_sys_stops['stop_id'].nunique()}")
        if(df_sys_stops.shape[0] == 0) : return
        
        df_sys_stops['type_stop'] = df_sys_stops[['home','work', 'other']].idxmax(axis=1)

        iter_sys_stops = zip(df_sys_stops['uid'], df_sys_stops['tid'], 
                             df_sys_stops['lat'], df_sys_stops['lng'], 
                             df_sys_stops['type_stop'],
                             df_sys_stops['stop_id'],
                             df_sys_stops['start_time'], df_sys_stops['end_time'])
        
        
        for uid, tid, lat, lng, type_stop, stop_id, start_hour, end_hour in iter_sys_stops :
    
            #print(f"{uid} -- {tid} -- {type_stop} -- {lat} -- {lng} -- {start_hour} -- {end_hour}")

            # Find the nodes in the graph associated with the "uid" and "tid" identifiers.
            user, traj, raw_traj = self.find_trajectories_from_graph(uid, tid)

            # *** Now, create all the triples neeed to semantically enrich the trajectory with this move. *** #
            # Feature node.
            URI_feat = 'http://example.org/user_' + str(uid) + '/traj_' + str(tid) + '/feature_sys_stop/'
            feature = URIRef(URI_feat)
            self.g.add((feature, RDF.type, self.STEP.Feature))
            self.g.add((traj, self.STEP.hasFeature, feature))

            # Episode node.
            URI_episode = URI_feat + str(stop_id) + '/'
            episode = URIRef(URI_episode)
            self.g.add((episode, RDF.type, self.STEP.Episode))
            self.g.add((feature, self.STEP.hasEpisode, episode))

            # Semantic description node.
            stop_desc = URIRef(URI_episode + 'desc/')
            self.g.add((stop_desc, RDF.type, self.dic_sys_stop[type_stop]))
            self.g.add((stop_desc, RDFS.subClassOf, self.STEP.Stop))
            self.g.add((stop_desc, self.STEP.hasStartHour, Literal(int(start_hour))))
            self.g.add((stop_desc, self.STEP.hasEndHour, Literal(int(end_hour))))
            self.g.add((episode, self.STEP.hasSemanticDescription, stop_desc))

            # Spatial extent (i.e, the latitude and longitude retrieved from the dataframe).
            sp_extent = URIRef(URI_episode + 'extent/')
            self.g.add((sp_extent, RDF.type, self.STEP.Point))           
            self.g.add((sp_extent, self.GEO.lat, Literal(lat)))
            self.g.add((sp_extent, self.GEO.long, Literal(lng)))
            self.g.add((episode, self.STEP.hasExtent, sp_extent))
            
            
            
    def add_moves(self, df_moves) :
        
        df_moves['datetime'] = pd.to_datetime(df_moves['datetime'], utc = True)

        #print(df_moves)
        #print(df_moves.info())
        
        # Compute a groupby on the enriched moves in order to extract the relevant information.
        res_gb = df_moves.groupby(['tid', 'move_id']).agg({'datetime': ["min", "max"], 'label': 'first', "uid": 'first'}).reset_index()
        # print(res_gb)
        iter_res = zip(res_gb[('uid', 'first')], res_gb['tid'], res_gb['move_id'], res_gb[('label', 'first')],
                       res_gb[('datetime', 'min')], res_gb[('datetime', 'max')])
        
        
        for uid, tid, move_id, type_move, start_t, end_t in iter_res :
    
            # print(f"{uid} -- {tid} -- {move_id} -- {type_move} -- {start} -- {end}")

            # Find the nodes in the graph associated with the "uid" and "tid" identifiers.
            user, traj, raw_traj = self.find_trajectories_from_graph(uid, tid)

            # Find the nodes in the graph corresponding to the instants and locations associated with this move.
            instant_start, location_start, instant_end, location_end = \
                self.find_instants_locations_start_end(raw_traj, start_t, end_t)


            # *** Now, create all the triples neeed to semantically enrich the trajectory with this move. *** #
            # Feature node.
            URI_feat = 'http://example.org/user_' + str(uid) + '/traj_' + str(tid) + '/feature_move/'
            feature = URIRef(URI_feat)
            self.g.add((feature, RDF.type, self.STEP.Feature))
            self.g.add((traj, self.STEP.hasFeature, feature))

            # Episode node.
            URI_episode = URI_feat + str(move_id) + '/'
            episode = URIRef(URI_episode)
            self.g.add((episode, RDF.type, self.STEP.Episode))
            self.g.add((feature, self.STEP.hasEpisode, episode))

            # Semantic description node.
            move_desc = URIRef(URI_episode + 'desc/')
            self.g.add((move_desc, RDF.type, self.dic_moves[type_move]))
            self.g.add((move_desc, RDFS.subClassOf, self.STEP.Move))
            self.g.add((episode, self.STEP.hasSemanticDescription, move_desc))

            # Spatiotemporal extent.
            st_extent = URIRef(URI_episode + 'extent/')
            self.g.add((st_extent, RDF.type, self.STEP.SpatiotemporalExtent))
            self.g.add((episode, self.STEP.hasExtent, st_extent))

            # Starting keypoint to associate to the spatiotemporal extent.
            start_kp = URIRef(URI_episode + 'extent/skp/')
            self.g.add((start_kp, RDF.type, self.STEP.KeyPoint))
            self.g.add((start_kp, self.STEP.atTime, instant_start))
            self.g.add((start_kp, self.STEP.hasLocation, location_start))
            self.g.add((st_extent, self.STEP.hasStartingPoint, start_kp))

            # Ending keypoint to associate to the spatiotemporal extent.
            end_kp = URIRef(URI_episode + 'extent/ekp/')
            self.g.add((end_kp, RDF.type, self.STEP.KeyPoint))
            self.g.add((end_kp, self.STEP.atTime, instant_end))
            self.g.add((end_kp, self.STEP.hasLocation, location_end))
            self.g.add((st_extent, self.STEP.hasEndingPoint, end_kp))
            
    
    def add_weather(self, weather_info) :
        
        print('Adding weather information to the RDF graph...')
        iter_rows = zip(weather_info['uid'], weather_info['tid'],
                        weather_info['lat'], weather_info['end_lat'],
                        weather_info['lng'], weather_info['end_lng'],
                        weather_info['datetime'], weather_info['end_datetime'],
                        weather_info['TAVG_C'], weather_info['DESCRIPTION'])
        
        id_weather = 0
        for uid, tid, lat_start, lat_end, lng_start, lng_end, t_start, t_end, temp, desc in iter_rows :
            
            # Find the nodes in the graph associated with the "uid" and "tid" identifiers.
            user, traj, raw_traj = self.find_trajectories_from_graph(uid, tid)
            
            # *** Now, create all the triples neeed to semantically enrich the trajectory with this move. *** #
            # Feature node.
            URI_feat = 'http://example.org/user_' + str(uid) + '/traj_' + str(tid) + '/feature_weather/'
            feature =  URIRef(URI_feat)
            self.g.add((feature, RDF.type, self.STEP.Feature))
            self.g.add((traj, self.STEP.hasFeature, feature))
            
            # Episode node.
            URI_episode = URI_feat + str(id_weather) + '/'
            episode = URIRef(URI_episode)
            self.g.add((episode, RDF.type, self.STEP.Episode))
            self.g.add((feature, self.STEP.hasEpisode, episode))
            
            # Semantic description node.
            weather_desc = URIRef(URI_episode + 'desc/')
            self.g.add((weather_desc, RDF.type, self.STEP.Weather))
            self.g.add((weather_desc, self.STEP.hasTemperature, Literal(temp)))
            self.g.add((weather_desc, self.STEP.hasWeatherCondition, Literal(desc)))
            self.g.add((episode, self.STEP.hasSemanticDescription, weather_desc))
            
            
            # *** Spatiotemporal extent *** #
            st_extent = URIRef(URI_episode + 'extent/')
            self.g.add((st_extent, RDF.type, self.STEP.SpatiotemporalExtent))
            self.g.add((episode, self.STEP.hasExtent, st_extent))
            
            # 1 - Starting keypoint to associate to the spatiotemporal extent.
            start_kp = URIRef(URI_episode + 'extent/skp/')
            self.g.add((start_kp, RDF.type, self.STEP.KeyPoint))
            self.g.add((st_extent, self.STEP.hasStartingPoint, start_kp))
            
            # Associate coordinates to the starting keypoint.
            start_point = URIRef(URI_episode + 'extent/skp/p/')
            self.g.add((start_point, RDF.type, self.STEP.Point))
            self.g.add((start_kp, self.STEP.hasLocation, start_point))
            self.g.add((start_point, self.GEO.lat, Literal(lat_start)))
            self.g.add((start_point, self.GEO.long, Literal(lng_start)))

            # Associate time instant to the starting keypoint.
            start_instant = URIRef(URI_episode + 'extent/skp/i/')
            self.g.add((start_instant, RDF.type, self.STEP.Instant))
            self.g.add((start_kp, self.STEP.atTime, start_instant))
            self.g.add((start_instant, TIME.inXSDDateTime, Literal(t_start)))

            
            # 2 - Ending keypoint to associate to the spatiotemporal extent.
            end_kp = URIRef(URI_episode + 'extent/ekp/')
            self.g.add((end_kp, RDF.type, self.STEP.KeyPoint))
            self.g.add((st_extent, self.STEP.hasEndingPoint, end_kp))
            
            # Associate coordinates to the starting keypoint.
            end_point = URIRef(URI_episode + 'extent/ekp/p/')
            self.g.add((end_point, RDF.type, self.STEP.Point))
            self.g.add((end_kp, self.STEP.hasLocation, end_point))
            self.g.add((end_point, self.GEO.lat, Literal(lat_end)))
            self.g.add((end_point, self.GEO.long, Literal(lng_end)))

            # Associate time instant to the starting keypoint.
            end_instant = URIRef(URI_episode + 'extent/ekp/i/')
            self.g.add((end_instant, RDF.type, self.STEP.Instant))
            self.g.add((end_kp, self.STEP.atTime, end_instant))
            self.g.add((end_instant, TIME.inXSDDateTime, Literal(t_end)))
            
            id_weather = id_weather + 1
            
            
            
    def add_social(self, social_info) :
        
        print('Adding social media posts of users to the RDF graph...')
        
        
        social_df = social_info.copy()
        social_df['uid'] = social_df['uid'].astype(str)
        social_df['tweet_created'] = pd.to_datetime(social_df['tweet_created'], utc = True)
        iter_rows = zip(social_df['uid'], social_df['tweet_created'], social_df['text'])
        
        
        id_post = 0
        for uid, time, text in iter_rows :
            
            # Find the Agent node whose foaf:name is equal to the "uid" identifier.
            # NOTE: a user may be missing from those that have at least a trajectory...in such case we skip the tweet.
            user_node = self.find_user_from_graph(uid)
            if user_node is None : continue
                
            # print(f"Adding tweet for user {user_node}")
            
            
            # *** Now, create all the triples neeed to semantically enrich the users with this post. *** #
            
            # Feature node.
            URI_feat = 'http://example.org/user_' + str(uid) + '/feature_social/'
            feature =  URIRef(URI_feat)
            self.g.add((feature, RDF.type, self.STEP.Feature))
            self.g.add((user_node, self.STEP.hasFeature, feature))
            
            # Episode node.
            URI_episode = URI_feat + str(id_post) + '/'
            episode = URIRef(URI_episode)
            self.g.add((episode, RDF.type, self.STEP.Episode))
            self.g.add((feature, self.STEP.hasEpisode, episode))
            
            # Semantic description node.
            social_desc = URIRef(URI_episode + 'desc/')
            self.g.add((social_desc, RDF.type, self.STEP.SocialMediaPost))
            self.g.add((social_desc, self.STEP.hasText, Literal(text)))
            self.g.add((episode, self.STEP.hasSemanticDescription, social_desc))
            
            # *** Temporal extent *** #
            t_extent = URIRef(URI_episode + 'extent/')
            self.g.add((t_extent, RDF.type, self.STEP.Instant))
            self.g.add((t_extent, TIME.inXSDDateTime, Literal(time)))
            self.g.add((episode, self.STEP.hasExtent, t_extent))
          
            id_post = id_post + 1
        
        
                    
    def serialize_graph(self, path, formato = 'turtle') :
        
        self.g.serialize(destination = path, format = formato)