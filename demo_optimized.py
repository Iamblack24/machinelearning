#!/usr/bin/env python3
"""
Demonstration of the optimized machine learning agent with performance monitoring.
"""

import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from learner import SelfLearningAgent
from performance_monitor import BottleneckAnalyzer, PerformanceMonitor, ResourceOptimizer
from logging_config import setup_optimized_logging, get_log_statistics, cleanup_old_logs


def demo_basic_functionality():
    """Demonstrate basic learning agent functionality."""
    print("=== Basic Functionality Demo ===")
    
    # Setup optimized logging
    setup_optimized_logging()
    
    # Initialize the agent
    agent = SelfLearningAgent()
    
    # Test mathematical operations
    print("\n1. Mathematical Operations:")
    math_tests = [
        "2 + 3 = 5",
        "10 - 4 = 6", 
        "7 * 8 = 56",
        "15 / 3 = 5"
    ]
    
    for test in math_tests:
        result = agent.interact(test)
        confidence = result.get('confidence', 0)
        print(f"   {test} → {result.get('type')} (confidence: {confidence:.1f})")
    
    # Test general knowledge
    print("\n2. General Knowledge:")
    knowledge_tests = [
        "Python is a programming language",
        "Machine learning uses algorithms",
        "Neural networks are inspired by the brain"
    ]
    
    for test in knowledge_tests:
        result = agent.interact(test)
        confidence = result.get('confidence', 0)
        print(f"   '{test}' → {result.get('type')} (confidence: {confidence:.1f})")
    
    # Test querying
    print("\n3. Knowledge Querying:")
    queries = [
        "What is 2 + 3?",
        "Tell me about Python",
        "machine learning"
    ]
    
    for query in queries:
        results = agent.query(query)
        print(f"   Query: '{query}' → {len(results)} results")
        if results:
            for i, result in enumerate(results[:2], 1):
                confidence = result.get('base_result', {}).get('confidence', 0)
                print(f"      {i}. Confidence: {confidence:.2f}")


def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n=== Performance Monitoring Demo ===")
    
    agent = SelfLearningAgent()
    
    # Perform some operations to generate data
    print("Generating sample interactions...")
    for i in range(20):
        agent.interact(f"Sample knowledge entry {i}")
        agent.interact(f"{i} + {i+1} = {i*2+1}")
    
    # Get memory usage statistics
    memory_stats = agent.get_memory_usage()
    print(f"\nMemory Usage:")
    print(f"   Entries: {memory_stats['knowledge_entries']}")
    print(f"   Estimated Usage: {memory_stats['estimated_usage_mb']} MB")
    print(f"   Within Limits: {memory_stats['within_limits']}")
    print(f"   Storage Utilization: {memory_stats.get('storage_utilization', 0):.1%}")
    
    # Run performance optimization
    print("\nRunning Performance Optimization...")
    optimization_result = agent.optimize_performance()
    print(f"   Actions taken: {optimization_result['actions_taken']}")
    print(f"   Performance improvement: {optimization_result['performance_improvement']:.1f}%")


def demo_bottleneck_analysis():
    """Demonstrate bottleneck analysis."""
    print("\n=== Bottleneck Analysis Demo ===")
    
    agent = SelfLearningAgent()
    analyzer = BottleneckAnalyzer()
    
    # Add some data to analyze
    for i in range(100):
        agent.interact(f"Test data {i}: This is a longer piece of content that simulates real usage patterns")
        if i % 20 == 0:
            agent.interact(f"{i} * 2 = {i*2}")
    
    # Analyze system
    analysis = analyzer.analyze_system(agent)
    
    print(f"Performance Score: {analysis['performance_score']:.1f}/100")
    
    if analysis['bottlenecks']:
        print("\nBottlenecks Found:")
        for bottleneck in analysis['bottlenecks']:
            print(f"   • {bottleneck}")
    else:
        print("\nNo significant bottlenecks detected!")
    
    if analysis['recommendations']:
        print("\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"   • {rec}")


def demo_resource_optimization():
    """Demonstrate resource optimization."""
    print("\n=== Resource Optimization Demo ===")
    
    agent = SelfLearningAgent()
    optimizer = ResourceOptimizer(target_memory_mb=4000)
    
    # Add duplicate and low-confidence data
    for i in range(50):
        agent.interact("This is duplicate content")  # Duplicates
        agent.knowledge_base.store(
            f"low_conf_{i}", 
            f"Low confidence data {i}",
            {'confidence': 0.1}  # Very low confidence
        )
    
    print(f"Before optimization: {len(agent.knowledge_base.knowledge)} entries")
    
    # Run optimization
    optimization_results = optimizer.optimize_knowledge_storage(agent.knowledge_base)
    
    print(f"After optimization: {optimization_results['final_entries']} entries")
    print(f"Memory saved estimate: {optimization_results['memory_saved_estimate']} bytes")
    
    if optimization_results['optimizations_applied']:
        print("Optimizations applied:")
        for opt in optimization_results['optimizations_applied']:
            print(f"   • {opt}")


def demo_logging_improvements():
    """Demonstrate logging improvements."""
    print("\n=== Logging Improvements Demo ===")
    
    # Setup optimized logging with small file size for demo
    log_file = setup_optimized_logging(max_file_size_mb=1, backup_count=3)
    print(f"Optimized logging configured: {log_file}")
    
    # Get log statistics
    log_stats = get_log_statistics()
    print(f"\nLog Statistics:")
    print(f"   Total log files: {log_stats['total_log_files']}")
    print(f"   Total size: {log_stats['total_size_mb']:.2f} MB")
    print(f"   Cleanup needed: {log_stats['needs_cleanup']}")
    
    # Cleanup old logs
    print("\nCleaning up old logs...")
    cleanup_old_logs(days_to_keep=1)  # Aggressive cleanup for demo


def main():
    """Run all demonstrations."""
    print("Machine Learning Agent - Optimized Performance Demo")
    print("=" * 50)
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        demo_basic_functionality()
        demo_performance_monitoring()
        demo_bottleneck_analysis()
        demo_resource_optimization()
        demo_logging_improvements()
        
        print("\n" + "=" * 50)
        print("✅ All demonstrations completed successfully!")
        print("\nKey improvements implemented:")
        print("   • Clean requirements.txt without conflicts")
        print("   • Consolidated SelfLearningAgent implementation")
        print("   • Memory-efficient knowledge storage with LRU optimization")
        print("   • Automatic performance monitoring and optimization")
        print("   • Rotating logs to prevent large log files")
        print("   • Resource management with 4GB memory target")
        print("   • Mathematical operations with proper validation")
        print("   • Bottleneck analysis and recommendations")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()