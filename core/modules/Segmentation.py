import pandas as pd
import numpy as np

import skmob
from skmob.preprocessing import detection

from core.ModuleInterface import ModuleInterface


class Segmentation(ModuleInterface):
    '''
    'Segmentation' models a class that segments a dataset of trajectories according to the stop and move paradigm.
    '''

    ### CLASS PUBLIC STATIC FIELDS ###

    id_class = 'Segmentation'



    ### PUBLIC CLASS CONSTRUCTOR ###
    
    def __init__(self) :

        print(f"Executing constructor of class {self.id_class}!")
        self.reset_state()
        
        

    ### CLASS PUBLIC METHODS ###
    def execute(self, dic_params: dict) -> bool:
        """
        This method executes the task logic associated with the Segmentation module.

        Parameters
        ----------
        dic_params : dict
            Dictionary that provides the input required by the module to execute its internal task logic.
            The dictionary contains (key,value) pairs, where key is the name of a specific input parameter and value
            the value passed for that input parameter.
            The input parameters that must be passed within 'dic_params' are:
                - 'trajectories': pandas DataFrame containing the trajectory dataset.
                - 'duration': int value specifying the minimum duration of a stop.
                - 'radius': float value specifying the maximum radius a stop can have.

        Returns
        -------
            execution_status : bool
                'True' if the execution went well, 'False' otherwise.
        """

        # Salva nei campi dell'istanza l'input passato
        self.trajectories = dic_params['trajectories']
        self.duration = dic_params['duration']
        self.radius = dic_params['radius']


        # Esegui il codice core dell'istanza.
        self.stops = None
        self.moves = None
        return self.core()


    def core(self) -> bool:

        # Load the trajectories into a skmob's TrajDataFrame, which in turn allows to perform the stop and move detection.
        tdf = skmob.TrajDataFrame(self.trajectories)


        ### stop detection ###
        stdf = detection.stay_locations(tdf,
                                        stop_radius_factor = 0.5,
                                        minutes_for_a_stop = self.duration,
                                        spatial_radius_km = self.radius,
                                        leaving_time = True)
        self.stops = pd.DataFrame(stdf)


        ### move detection ###
        trajs = tdf.copy()
        starts = stdf.copy()
        ends = stdf.copy()

        trajs.set_index(['tid','datetime'], inplace = True)
        starts.set_index(['tid','datetime'], inplace = True)
        ends.set_index(['tid','leaving_datetime'], inplace = True)

        traj_ids = trajs.index
        start_ids = starts.index
        end_ids = ends.index

        # some datetime into stdf are approximated. In order to retrieve moves, we have to check the exact datime into 
        # trajectory dataframe. We use `isin()` method to reduce time computation
        traj_df = pd.DataFrame(traj_ids, columns=['trajs'])
        start_df = pd.DataFrame(start_ids, columns=['start'])
        end_df = pd.DataFrame(end_ids, columns=['end'])

        start_df['is_in_traj'] = start_df['start'].isin(traj_df['trajs'])
        end_df['is_in_traj'] = end_df['end'].isin(traj_df['trajs'])

        start_df['end'] = end_df['end']
        start_df['is_in_traj_end'] = end_df['is_in_traj']

        # remove stops which aren't into tdf
        start_df = start_df[(start_df['is_in_traj']!=False)|(start_df['is_in_traj_end']!=False)]

        # save index of incomplete stops and convert them into MultiIndex
        incomplete_end = start_df['end'][(start_df['is_in_traj']==False)&(start_df['is_in_traj_end']==True)] 
        incomplete_start = start_df['start'][(start_df['is_in_traj']==True)&(start_df['is_in_traj_end']==False)]

        if not incomplete_end.empty:
            incomplete_end = pd.MultiIndex.from_tuples(incomplete_end)

        if not incomplete_start.empty:
            incomplete_start = pd.MultiIndex.from_tuples(incomplete_start)

        # save complete index
        start_df = start_df[(start_df['is_in_traj']==True)&(start_df['is_in_traj_end']==True)] 
        
        new_start = pd.MultiIndex.from_tuples(start_df['start'])
        new_end = pd.MultiIndex.from_tuples(start_df['end'])
        new_start.set_names(['tid','datetime'],inplace=True)
        new_end.set_names(['tid','datetime'],inplace=True)
        
        # set start and end of stops (using two columns in order to avoid overlaps)
        trajs['start_stop'] = np.nan
        trajs['start_stop'].loc[new_start] = 1
        trajs['end_stop'] = np.nan
        trajs['end_stop'].loc[new_end] = 1

        trajs.reset_index(inplace=True)
        start_idx = trajs[trajs['start_stop']==1].index.to_list()
        end_idx = trajs[trajs['end_stop']==1].index.to_list()

        # set incomplete index
        starts_ = [traj_ids.get_loc(e).start - 1 for e in incomplete_end]
        ends_ = [traj_ids.get_loc(s).start + 1 for s in incomplete_start]

        if starts_ != []:
            start_idx = start_idx + starts_
    
        if ends_ != []:
            end_idx = end_idx + ends_

        trajs['move_id'] = np.nan
        
        for i, (s, e) in enumerate(zip(start_idx,end_idx), 1):
            trajs['move_id'][s: e+1] = i


        trajs['move_id'].ffill(inplace=True)
        trajs['move_id'].fillna(0,inplace=True)
        trajs['move_id'][(trajs['start_stop']==1)|(trajs['end_stop']==1)] = -1
        moves = trajs[trajs['move_id']!=-1]

        # NOTE: the final moves result set will be a pandas DataFrame built from the skmob dataframe.
        moves.drop(columns = ['start_stop', 'end_stop'], inplace = True)
        moves['move_id'] = moves['move_id'].astype(np.uint32)
        self.moves = pd.DataFrame(moves)


        return True

    def get_results(self) -> dict :

        return {'trajectories' : self.trajectories.copy() if self.trajectories is not None else None,
                'stops' : self.stops.copy() if self.stops is not None else None,
                'moves' : self.moves.copy() if self.moves is not None else None}

    def get_params_input(self) -> list[str] :
        return ['trajectories', 'duration' 'radius']

    def get_params_output(self) -> list[str] :
        return list(self.get_results().keys())
            
    def reset_state(self) :
        self.trajectories = None
        self.stops = None
        self.moves = None
        self.radius = None
        self.duration = None