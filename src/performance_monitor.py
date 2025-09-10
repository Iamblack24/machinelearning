"""
Performance monitoring and optimization utilities for the machine learning system.
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional
from functools import wraps
from datetime import datetime


class PerformanceMonitor:
    """Monitor and track performance metrics for the learning system."""
    
    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
        
    def track_execution_time(self, func_name: str = None):
        """Decorator to track function execution time."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                execution_time = end_time - start_time
                name = func_name or func.__name__
                
                if name not in self.metrics:
                    self.metrics[name] = {
                        'total_calls': 0,
                        'total_time': 0.0,
                        'average_time': 0.0,
                        'min_time': float('inf'),
                        'max_time': 0.0
                    }
                
                metrics = self.metrics[name]
                metrics['total_calls'] += 1
                metrics['total_time'] += execution_time
                metrics['average_time'] = metrics['total_time'] / metrics['total_calls']
                metrics['min_time'] = min(metrics['min_time'], execution_time)
                metrics['max_time'] = max(metrics['max_time'], execution_time)
                
                if execution_time > 1.0:  # Log slow operations
                    self.logger.warning(f"Slow operation: {name} took {execution_time:.2f}s")
                
                return result
            return wrapper
        return decorator
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics.copy(),
            'summary': {
                'total_functions_tracked': len(self.metrics),
                'slowest_function': None,
                'fastest_function': None,
                'most_called_function': None
            }
        }
        
        if self.metrics:
            # Find slowest function by average time
            slowest = max(self.metrics.items(), key=lambda x: x[1]['average_time'])
            report['summary']['slowest_function'] = {
                'name': slowest[0],
                'average_time': slowest[1]['average_time']
            }
            
            # Find fastest function by average time
            fastest = min(self.metrics.items(), key=lambda x: x[1]['average_time'])
            report['summary']['fastest_function'] = {
                'name': fastest[0],
                'average_time': fastest[1]['average_time']
            }
            
            # Find most called function
            most_called = max(self.metrics.items(), key=lambda x: x[1]['total_calls'])
            report['summary']['most_called_function'] = {
                'name': most_called[0],
                'total_calls': most_called[1]['total_calls']
            }
        
        return report
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        self.metrics.clear()
        self.logger.info("Performance metrics reset")


class ResourceOptimizer:
    """Optimize resource usage for the learning system."""
    
    def __init__(self, target_memory_mb: int = 4000):
        self.target_memory_mb = target_memory_mb
        self.logger = logging.getLogger(__name__)
        
    def optimize_knowledge_storage(self, knowledge_base) -> Dict[str, Any]:
        """Optimize knowledge storage to reduce memory usage."""
        optimization_results = {
            'initial_entries': len(knowledge_base.knowledge),
            'optimizations_applied': [],
            'final_entries': 0,
            'memory_saved_estimate': 0
        }
        
        # Remove duplicate entries based on content similarity
        duplicates_removed = self._remove_duplicates(knowledge_base)
        if duplicates_removed > 0:
            optimization_results['optimizations_applied'].append(
                f"Removed {duplicates_removed} duplicate entries"
            )
        
        # Compress old entries with low confidence
        compressed = self._compress_low_confidence_entries(knowledge_base)
        if compressed > 0:
            optimization_results['optimizations_applied'].append(
                f"Compressed {compressed} low-confidence entries"
            )
        
        optimization_results['final_entries'] = len(knowledge_base.knowledge)
        optimization_results['memory_saved_estimate'] = (
            optimization_results['initial_entries'] - optimization_results['final_entries']
        ) * 1024  # Rough estimate in bytes
        
        return optimization_results
    
    def _remove_duplicates(self, knowledge_base) -> int:
        """Remove duplicate knowledge entries."""
        initial_count = len(knowledge_base.knowledge)
        seen_contents = set()
        keys_to_remove = []
        
        for key, entry in knowledge_base.knowledge.items():
            content = str(entry.get('content', ''))
            content_hash = hash(content)
            
            if content_hash in seen_contents:
                keys_to_remove.append(key)
            else:
                seen_contents.add(content_hash)
        
        # Remove duplicates
        for key in keys_to_remove:
            del knowledge_base.knowledge[key]
        
        removed_count = len(keys_to_remove)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} duplicate entries")
        
        return removed_count
    
    def _compress_low_confidence_entries(self, knowledge_base) -> int:
        """Compress or remove entries with very low confidence."""
        compressed_count = 0
        keys_to_modify = []
        
        for key, entry in knowledge_base.knowledge.items():
            confidence = entry.get('confidence', 0.5)
            
            # If confidence is very low and entry is old, compress it
            if confidence < 0.2:
                timestamp = entry.get('timestamp', '')
                if timestamp:
                    try:
                        entry_time = datetime.fromisoformat(timestamp)
                        age_hours = (datetime.now() - entry_time).total_seconds() / 3600
                        
                        if age_hours > 24:  # Older than 24 hours
                            keys_to_modify.append(key)
                    except ValueError:
                        pass  # Skip if timestamp is invalid
        
        # Compress entries by keeping only essential information
        for key in keys_to_modify:
            entry = knowledge_base.knowledge[key]
            compressed_entry = {
                'content': str(entry.get('content', ''))[:100] + "...",  # Truncate content
                'timestamp': entry.get('timestamp', ''),
                'confidence': entry.get('confidence', 0.0),
                'metadata': {'compressed': True, 'original_type': entry.get('metadata', {}).get('type', 'unknown')}
            }
            knowledge_base.knowledge[key] = compressed_entry
            compressed_count += 1
        
        if compressed_count > 0:
            self.logger.info(f"Compressed {compressed_count} low-confidence entries")
        
        return compressed_count


class BottleneckAnalyzer:
    """Analyze system bottlenecks and suggest improvements."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_system(self, agent) -> Dict[str, Any]:
        """Perform comprehensive bottleneck analysis."""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'bottlenecks': [],
            'recommendations': [],
            'performance_score': 0.0
        }
        
        # Analyze knowledge base size
        kb_analysis = self._analyze_knowledge_base(agent.knowledge_base)
        analysis['bottlenecks'].extend(kb_analysis['bottlenecks'])
        analysis['recommendations'].extend(kb_analysis['recommendations'])
        
        # Analyze memory usage
        memory_analysis = self._analyze_memory_usage(agent)
        analysis['bottlenecks'].extend(memory_analysis['bottlenecks'])
        analysis['recommendations'].extend(memory_analysis['recommendations'])
        
        # Calculate overall performance score
        bottleneck_count = len(analysis['bottlenecks'])
        analysis['performance_score'] = max(0.0, 100.0 - (bottleneck_count * 10))
        
        return analysis
    
    def _analyze_knowledge_base(self, knowledge_base) -> Dict[str, List[str]]:
        """Analyze knowledge base for potential bottlenecks."""
        analysis = {'bottlenecks': [], 'recommendations': []}
        
        entry_count = len(knowledge_base.knowledge)
        
        if entry_count > 10000:
            analysis['bottlenecks'].append(
                f"Large knowledge base with {entry_count} entries may slow queries"
            )
            analysis['recommendations'].append(
                "Consider implementing knowledge base pruning and indexing"
            )
        
        # Check for content size issues
        large_entries = 0
        for entry in knowledge_base.knowledge.values():
            content_size = len(str(entry.get('content', '')))
            if content_size > 10000:  # 10KB per entry
                large_entries += 1
        
        if large_entries > 100:
            analysis['bottlenecks'].append(
                f"{large_entries} entries are very large, consuming excessive memory"
            )
            analysis['recommendations'].append(
                "Implement content compression for large entries"
            )
        
        return analysis
    
    def _analyze_memory_usage(self, agent) -> Dict[str, List[str]]:
        """Analyze memory usage patterns."""
        analysis = {'bottlenecks': [], 'recommendations': []}
        
        memory_info = agent.get_memory_usage()
        
        if not memory_info['within_limits']:
            analysis['bottlenecks'].append(
                f"Memory usage exceeds limits: {memory_info['estimated_usage_mb']}MB"
            )
            analysis['recommendations'].append(
                "Implement aggressive memory optimization and garbage collection"
            )
        
        knowledge_entries = memory_info['knowledge_entries']
        if knowledge_entries > 5000:
            analysis['bottlenecks'].append(
                f"High number of knowledge entries ({knowledge_entries}) may impact performance"
            )
            analysis['recommendations'].append(
                "Implement knowledge base archiving for old entries"
            )
        
        return analysis


# Global performance monitor instance
performance_monitor = PerformanceMonitor()