from typing import Dict, Any, List

class Explainer:
    def __init__(self):
        self.decision_paths = {}

    def record_decision(self, decision_id: str, path: List[str], confidence: float):
        self.decision_paths[decision_id] = {
            'path': path,
            'confidence': confidence
        }

    def generate_explanation(self, decision_id: str) -> Dict[str, Any]:
        detail = self.decision_paths.get(decision_id)
        if detail:
            return {
                'explanation': f"Decision path: {' -> '.join(detail['path'])}",
                'confidence': detail['confidence']
            }
        return {'explanation': "No explanation available.", 'confidence': 0.0}

    def visualize_knowledge_path(self, decision_id: str) -> str:
        detail = self.decision_paths.get(decision_id)
        if detail:
            return f"[Visualization of: {' -> '.join(detail['path'])}]"
        return "No visualization available."