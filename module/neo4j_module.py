from typing import Literal,Any
from collections import defaultdict
from tqdm import tqdm
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

class Neo4j_Interface:
    def __init__(self,port:int=7687):
        self.connect_db(port=port)

    def connect_db(self,port:int=7687):
        try:
            self.driver=GraphDatabase.driver(
                f"neo4j://127.0.0.1:{port}",
                auth=("neo4j","11111111")
            )
            self.driver.verify_connectivity()
            print("Neo4j 연결 성공")
        except Neo4jError as e:
            print(f"Neo4j error: {e}")

    def disconnect_db(self):
        if self.driver is not None:
            self.driver.close()
            self.driver=None
            print("Neo4j database disconnected!")

    def delete_graph(self,
            graph_id:str
        ):
        """
        Neo4j의 특정 graph의 노드와 관계를 삭제
        """
        query=f"""
            MATCH (n {{graph_id: $graph_id}})
            DETACH DELETE n;
        """
        with self.driver.session() as session:
            session.run(query,graph_id=graph_id).consume()
        print(f"Delete graph elements: {graph_id}")

    def insert_nodes(self,
            nodes:list[dict]
        ):
        """
        """

        query="""
            UNWIND $nodes AS node

            MERGE (n:pin {
                graph_id: node.graph_id,
                pin_id: node.pin_id
            })

            SET
                n.x1=node.x1,
                n.y1=node.y1,
                n.x2=node.x2,
                n.y2=node.y2
        """