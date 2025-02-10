import networkx as nx
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import json
import uuid

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.version_history = {}
        
    def add_node(self, 
                 node_id: str, 
                 content: str, 
                 source: str,
                 confidence: float = 0.0,
                 metadata: Optional[Dict] = None) -> str:
        """
        Add a node to the knowledge graph with versioning and confidence scoring.
        
        Args:
            node_id: Unique identifier for the node
            content: The actual knowledge content
            source: Source of the knowledge
            confidence: Initial confidence score (0-1)
            metadata: Additional metadata
        """
        if metadata is None:
            metadata = {}
            
        timestamp = datetime.utcnow().isoformat()
        version_id = str(uuid.uuid4())
        
        node_data = {
            'content': content,
            'source': source,
            'confidence': confidence,
            'created_at': timestamp,
            'last_updated': timestamp,
            'version_id': version_id,
            'previous_versions': [],
            'metadata': metadata
        }
        
        # Store version history
        self.version_history[version_id] = node_data.copy()
        
        # Add to graph
        self.graph.add_node(node_id, **node_data)
        return node_id

    def add_relationship(self, 
                        source_id: str, 
                        target_id: str, 
                        relationship_type: str,
                        confidence: float = 0.0,
                        metadata: Optional[Dict] = None) -> None:
        """
        Add a relationship between two nodes.
        """
        if metadata is None:
            metadata = {}
            
        self.graph.add_edge(
            source_id, 
            target_id, 
            relationship_type=relationship_type,
            confidence=confidence,
            created_at=datetime.utcnow().isoformat(),
            metadata=metadata
        )
    
    def update_node(self, 
                    node_id: str, 
                    content: str, 
                    confidence: Optional[float] = None,
                    metadata: Optional[Dict] = None) -> str:
        """
        Update a node while maintaining version history.
        """
        if node_id not in self.graph:
            raise KeyError(f"Node {node_id} not found in graph")
            
        old_data = self.graph.nodes[node_id]
        new_version_id = str(uuid.uuid4())
        
        # Create new version
        new_data = old_data.copy()
        new_data['content'] = content
        new_data['last_updated'] = datetime.utcnow().isoformat()
        new_data['previous_versions'].append(old_data['version_id'])
        new_data['version_id'] = new_version_id
        
        if confidence is not None:
            new_data['confidence'] = confidence
        if metadata:
            new_data['metadata'].update(metadata)
            
        # Store version history
        self.version_history[new_version_id] = new_data.copy()
        
        # Update graph
        self.graph.nodes[node_id].update(new_data)
        return new_version_id
    
    def get_node_history(self, node_id: str) -> List[Dict]:
        """
        Get the version history of a node.
        """
        if node_id not in self.graph:
            raise KeyError(f"Node {node_id} not found in graph")
            
        node_data = self.graph.nodes[node_id]
        history = [self.version_history[node_data['version_id']]]
        
        for version_id in node_data['previous_versions']:
            history.append(self.version_history[version_id])
            
        return history
    
    def validate_knowledge(self, node_id: str) -> Tuple[bool, float]:
        """
        Validate knowledge based on relationships and confidence scores.
        """
        if node_id not in self.graph:
            raise KeyError(f"Node {node_id} not found in graph")
            
        # Get all incoming edges (supporting evidence)
        predecessors = self.graph.predecessors(node_id)
        confidence_scores = [
            self.graph.nodes[pred]['confidence'] 
            for pred in predecessors
        ]
        
        # Simple validation based on average confidence of related nodes
        if not confidence_scores:
            return True, self.graph.nodes[node_id]['confidence']
            
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        is_valid = avg_confidence >= 0.5
        
        return is_valid, avg_confidence
    
    def export_to_json(self, filepath: str) -> None:
        """
        Export the knowledge graph to JSON format.
        """
        data = {
            'nodes': dict(self.graph.nodes(data=True)),
            'edges': dict(self.graph.edges(data=True)),
            'version_history': self.version_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'KnowledgeGraph':
        """
        Load a knowledge graph from JSON format.
        """
        kg = cls()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Reconstruct graph
        for node_id, node_data in data['nodes'].items():
            kg.graph.add_node(node_id, **node_data)
            
        for source, target, edge_data in data['edges'].items():
            kg.graph.add_edge(source, target, **edge_data)
            
        kg.version_history = data['version_history']
        return kg