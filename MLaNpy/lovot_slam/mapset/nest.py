from typing import Optional

from lovot_slam.redis.clients import create_ltm_client


class NestMapsetPair:
    """
    Manager for nest and mapset id pair
        this is a dictionary that stores the pair of nest id and mapset id
        (hash -> key: nest id / value: mapset id)
    """
    _KEY = 'slam:nest_map_ids_pair'  
    _ltm = create_ltm_client() 

    @classmethod
    def query_nest_mapset_pair(cls, nest_id:str) -> Optional[str]:
        """
        Args:
            redis (Redis): redis instance
            nest_id (str): nest id
        Return:
            (str): mapset id / None
        """
        mapset_id = cls._ltm.hget(cls._KEY, nest_id)

        return mapset_id

    @classmethod
    def add_nest_mapset_pair(cls, nest_id:str, mapset_id:str) -> None:
        """
        Args:
            redis (Redis): redis instance
            nest_id (str): nest id
            mapset_id (str): mapset id
        """
        cls._ltm.hset(cls._KEY, nest_id, mapset_id)

    @classmethod
    def remove_nest_mapset_pair(cls, nest_id:str) -> None:
        """
        Args:
            redis (Redis): redis instance
            nest_id (str): nest id
        """
        cls._ltm.hdel(cls._KEY, nest_id)
        
    @classmethod
    def remove_all(cls) -> None:
        """
        Args:
            redis (Redis): redis instance
        """
        cls._ltm.delete(cls._KEY)