import geopandas as gpd
import pandas as pd

from rdflib import Graph, Namespace
from rdflib import Literal, URIRef, BNode
from rdflib import RDF
from rdflib.namespace import FOAF, TIME


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
        # self.STEP = Namespace('http://purl.org/net/step#')
        self.STEP = Namespace('http://purl.org/net/step_specialized#')
        self.GEO = Namespace('http://www.w3.org/2003/01/geo/wgs84_pos#')


        # Bind the above references to the ontologies to the graph.
        self.g.bind('step_specialized', self.STEP)
        self.g.bind('geo', self.GEO)
        
        
        # Define a dictionary containing the mapping to be used for transport mode.
        self.dic_sys_stop = {'home' : self.STEP.Home,
                             'work' : self.STEP.Work,
                             'other' : self.STEP.Regular_Stop}
        
        
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
        print(df_raw_traj)
        print(df_raw_traj.info())


        # Retrieve the list of users from the dataframe.
        list_users_cleaned = df_raw_traj['uid'].unique()
        print(f"List of unique user IDs: {list_users_cleaned}")
        
        
        # *** Build the part of the KG related to the users and their trajectories. *** #
        for t in list_users_cleaned :
        
            view = df_raw_traj.loc[df_raw_traj['uid'] == t]
            #display(view)
            
            # Create the agent node.
            agent = BNode()
            print(f"Considering user {t} (graph node with ID {agent})")
            
            self.g.add((agent, RDF.type, FOAF.Agent))
            self.g.add((agent, FOAF.name, Literal(t)))
            
            # Associate to the agent all its trajectories
            list_sub_t = view["tid"].unique()
            # print(f"List of subtrajectory IDs associated with the user {t}: {list_sub_t}")
            print(f"Number of trajectories associated with the user {t}: {len(list_sub_t)}")
            print(f"Number of samples associated with the user {t}: {view.shape[0]}")
            
            for s in list_sub_t :
                
                # Create the "logical" trajectory which will hold the raw trajectory + its semantic aspects.
                traj = BNode()
                self.g.add((agent, self.STEP.hasTrajectory , traj))
                self.g.add((traj, RDF.type, self.STEP.Trajectory))
                self.g.add((traj, self.STEP.hasID, Literal(s))) # Save the ID of this trajectory.
                
                
                # Create the raw trajectory
                seg = BNode()
                self.g.add((seg, RDF.type, self.STEP.RawTrajectory))
                self.g.add((traj, self.STEP.hasRawTrajectory, seg))
                
                
                # Extract the info associated with the fixes.
                view_2 = view.loc[view['tid'] == s]
                list_lat = view_2['lat']
                list_long = view_2['lng']
                list_time = view_2['datetime']
                list_fix = zip(list_lat, list_long, list_time)
                # print(f"Generating trajectory {s} for user {t} (samples: {len(view_2)})")
                
                for (lat, long, time) in list_fix :
                    fix = BNode()
                    point = BNode()
                    instant = BNode()
                    
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
                    
                    
    def add_occasional_stops(self, df_occasional_stops) :
        
        df_occasional_stops['datetime'] = pd.to_datetime(df_occasional_stops['datetime'], utc = True)
        df_occasional_stops['leaving_datetime'] = pd.to_datetime(df_occasional_stops['leaving_datetime'], utc = True)
        #display(df_occasional_stops.info())

        view_stop_data = df_occasional_stops[['stop_id', 'uid', 'tid', 'datetime', 'leaving_datetime', 'osmid']]
        #display(view_stop_data)
        print(f"Number of occasional stops: {view_stop_data['stop_id'].nunique()}")
        
        gb = view_stop_data.groupby(['stop_id'])
        for key in gb.groups.keys() :

            # Retrieve the rows associated with this stop.
            group = gb.get_group(key)

            # Retrieve the main properties of this stop.
            uid = group['uid'].iloc[0]
            tid = group['tid'].iloc[0]
            start = group['datetime'].iloc[0]
            end = group['leaving_datetime'].iloc[0]
            list_POI = group['osmid']

            #print(f"{key} -- {uid} -- {tid} -- {start} -- {end}")
            #display(list_POI)
            #break

            # Find the nodes in the graph associated with the "uid" and "tid" identifiers.
            user, traj, raw_traj = self.find_trajectories_from_graph(uid, tid)

            # Find the nodes in the graph corresponding to the instants and locations associated with this move.
            instant_start, location_start, instant_end, location_end = \
                self.find_instants_locations_start_end(raw_traj, start, end)


            # *** Now, create all the triples neeed to semantically enrich the trajectory with this move. *** #
            # Feature node.
            feature = BNode()
            self.g.add((feature, RDF.type, self.STEP.Feature))
            self.g.add((traj, self.STEP.hasFeature, feature))

            # Episode node.
            episode = BNode()
            self.g.add((episode, RDF.type, self.STEP.Episode))
            self.g.add((feature, self.STEP.hasEpisode, episode))

            # Semantic description (Occasional Stop) node.
            stop_desc = BNode()
            self.g.add((stop_desc, RDF.type, self.STEP.Occasional_Stop))
            self.g.add((episode, self.STEP.hasSemanticDescription, stop_desc))

            # Associate with the Occasional Stop all the POIs that may be associated with it.
            # Passano alcuni valori NaN, da controllare!
            for p in list_POI :
                poi = BNode()
                self.g.add((poi, RDF.type, self.STEP.Point_of_Interest))
                self.g.add((poi, self.STEP.hasOSMValue, Literal(str(p))))
                self.g.add((stop_desc, self.STEP.hasPOI, poi))

            # Spatiotemporal extent.
            st_extent = BNode()
            self.g.add((st_extent, RDF.type, self.STEP.SpatiotemporalExtent))
            self.g.add((episode, self.STEP.hasExtent, st_extent))

            # Starting keypoint to associate to the spatiotemporal extent.
            start_kp = BNode()
            self.g.add((start_kp, RDF.type, self.STEP.KeyPoint))
            self.g.add((start_kp, self.STEP.atTime, instant_start))
            self.g.add((start_kp, self.STEP.hasLocation, location_start))
            self.g.add((st_extent, self.STEP.hasStartingPoint, start_kp))

            # Ending keypoint to associate to the spatiotemporal extent.
            end_kp = BNode()
            self.g.add((end_kp, RDF.type, self.STEP.KeyPoint))
            self.g.add((end_kp, self.STEP.atTime, instant_end))
            self.g.add((end_kp, self.STEP.hasLocation, location_end))
            self.g.add((st_extent, self.STEP.hasStartingPoint, end_kp))
            
            
    def add_systematic_stops(self, df_sys_stops) :
        
        df_sys_stops['type_stop'] = df_sys_stops[['home','work', 'other']].idxmax(axis=1)
        print(df_sys_stops)
        print(df_sys_stops.info())

        print(f"Number of systematic stops: {df_sys_stops['stop_id'].nunique()}")

        iter_sys_stops = zip(df_sys_stops['uid'], df_sys_stops['tid'], 
                             df_sys_stops['lat'], df_sys_stops['lng'], 
                             df_sys_stops['type_stop'],
                             df_sys_stops['start_time'], df_sys_stops['end_time'])
        
        
        for uid, tid, lat, lng, type_stop, start_hour, end_hour in iter_sys_stops :
    
            #print(f"{uid} -- {tid} -- {type_stop} -- {lat} -- {lng} -- {start_hour} -- {end_hour}")

            # Find the nodes in the graph associated with the "uid" and "tid" identifiers.
            user, traj, raw_traj = self.find_trajectories_from_graph(uid, tid)

            # *** Now, create all the triples neeed to semantically enrich the trajectory with this move. *** #
            # Feature node.
            feature = BNode()
            self.g.add((feature, RDF.type, self.STEP.Feature))
            self.g.add((traj, self.STEP.hasFeature, feature))

            # Episode node.
            episode = BNode()
            self.g.add((episode, RDF.type, self.STEP.Episode))
            self.g.add((feature, self.STEP.hasEpisode, episode))

            # Semantic description node.
            stop_desc = BNode()
            self.g.add((stop_desc, RDF.type, self.dic_sys_stop[type_stop]))
            self.g.add((stop_desc, self.STEP.hasStartHour, Literal(start_hour)))
            self.g.add((stop_desc, self.STEP.hasEndHour, Literal(end_hour)))
            self.g.add((episode, self.STEP.hasSemanticDescription, stop_desc))

            # Spatial extent (i.e, the latitude and longitude retrieved from the dataframe).
            sp_extent = BNode()
            self.g.add((sp_extent, RDF.type, self.STEP.Point))           
            self.g.add((sp_extent, self.GEO.lat, Literal(lat)))
            self.g.add((sp_extent, self.GEO.long, Literal(lng)))
            self.g.add((episode, self.STEP.hasExtent, sp_extent))
            
            
            
    def add_moves(self, df_moves) :
        
        df_moves['datetime'] = pd.to_datetime(df_moves['datetime'], utc = True)

        print(df_moves)
        print(df_moves.info())
        print(f"Modes of movement: {df_moves['label'].unique()}")
        
        
        # Compute a groupby on the enriched moves in order to extract the relevant information.
        res_gb = df_moves.groupby(['tid', 'move_id']).agg({'datetime': ["min", "max"], 'label': 'first', "uid": 'first'}).reset_index()
        print(res_gb)
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
            feature = BNode()
            self.g.add((feature, RDF.type, self.STEP.Feature))
            self.g.add((traj, self.STEP.hasFeature, feature))

            # Episode node.
            episode = BNode()
            self.g.add((episode, RDF.type, self.STEP.Episode))
            self.g.add((feature, self.STEP.hasEpisode, episode))

            # Semantic description node.
            move_desc = BNode()
            self.g.add((move_desc, RDF.type, self.dic_moves[type_move]))
            self.g.add((episode, self.STEP.hasSemanticDescription, move_desc))

            # Spatiotemporal extent.
            st_extent = BNode()
            self.g.add((st_extent, RDF.type, self.STEP.SpatiotemporalExtent))
            self.g.add((episode, self.STEP.hasExtent, st_extent))

            # Starting keypoint to associate to the spatiotemporal extent.
            start_kp = BNode()
            self.g.add((start_kp, RDF.type, self.STEP.KeyPoint))
            self.g.add((start_kp, self.STEP.atTime, instant_start))
            self.g.add((start_kp, self.STEP.hasLocation, location_start))
            self.g.add((st_extent, self.STEP.hasStartingPoint, start_kp))

            # Ending keypoint to associate to the spatiotemporal extent.
            end_kp = BNode()
            self.g.add((end_kp, RDF.type, self.STEP.KeyPoint))
            self.g.add((end_kp, self.STEP.atTime, instant_end))
            self.g.add((end_kp, self.STEP.hasLocation, location_end))
            self.g.add((st_extent, self.STEP.hasStartingPoint, end_kp))
        
        
                    
    def serialize_graph(self, path, formato = 'turtle') :
        
        # self.g.serialize(destination = path, format = "pretty-xml")
        self.g.serialize(destination = path, format = formato)