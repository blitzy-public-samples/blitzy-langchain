"""Chain Introspection Utility for LangChain Debugging.

This module provides utilities for inspecting and analyzing LangChain chain structure
and composition. It helps developers understand how chains are constructed, visualize
their execution graph, and debug complex chain architectures.

Source: Agent Action Plan section 0.5.1, 0.8.1
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from langchain_core.runnables import Runnable
from langchain_core.runnables.base import RunnableSequence
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.graph import Graph, Node, Edge


def inspect_chain(
    chain: Runnable[Any, Any],
    config: Optional[RunnableConfig] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Inspect and print chain components and structure.

    Analyzes a LangChain Runnable to extract and display its internal structure,
    including component types, configuration, and composition patterns. Useful for
    understanding how a chain is constructed and debugging complex architectures.

    Args:
        chain: The Runnable chain to inspect. Can be any LangChain Runnable including
            simple chains, LCEL compositions, or complex agent executors.
        config: Optional RunnableConfig to use when inspecting the chain. This can
            affect how the graph is generated for configurable chains.
        verbose: If True, prints detailed information to stdout. If False, only
            returns the inspection data without printing.

    Returns:
        A dictionary containing the inspection results with the following keys:
            - "name": The name of the chain/runnable
            - "type": The class name of the chain
            - "has_graph": Boolean indicating if get_graph() is supported
            - "node_count": Number of nodes in the graph (if available)
            - "edge_count": Number of edges in the graph (if available)
            - "components": List of component names in the chain
            - "config_schema": JSON schema for configuration (if available)

    Example:
        >>> from langchain_core.prompts import ChatPromptTemplate
        >>> from langchain_core.runnables import RunnableLambda
        >>> 
        >>> # Create a simple chain
        >>> prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
        >>> chain = prompt | RunnableLambda(lambda x: x.to_string())
        >>> 
        >>> # Inspect the chain
        >>> info = inspect_chain(chain, verbose=True)
        >>> print(f"Chain has {info['node_count']} nodes")
    """
    result: Dict[str, Any] = {
        "name": chain.get_name() if hasattr(chain, "get_name") else str(type(chain).__name__),
        "type": type(chain).__name__,
        "has_graph": False,
        "node_count": 0,
        "edge_count": 0,
        "components": [],
        "config_schema": None,
    }

    if verbose:
        print("=" * 80)
        print(f"Chain Inspection: {result['name']}")
        print("=" * 80)
        print(f"Type: {result['type']}")
        print()

    # Try to get the graph representation
    try:
        graph = chain.get_graph(config=config)
        result["has_graph"] = True
        result["node_count"] = len(graph.nodes)
        result["edge_count"] = len(graph.edges)

        # Extract component names from graph nodes
        components = []
        for node in graph.nodes.values():
            if node.data and hasattr(node.data, "get_name"):
                components.append(node.data.get_name())
            elif node.data:
                components.append(str(type(node.data).__name__))
        result["components"] = components

        if verbose:
            print(f"Graph Support: Yes")
            print(f"Nodes: {result['node_count']}")
            print(f"Edges: {result['edge_count']}")
            print()
            print("Components:")
            for i, component in enumerate(components, 1):
                print(f"  {i}. {component}")
            print()

    except Exception as e:
        if verbose:
            print(f"Graph Support: No ({type(e).__name__}: {str(e)})")
            print()

    # Try to get config schema
    try:
        schema = chain.config_schema()
        if schema:
            result["config_schema"] = schema.model_json_schema()
            if verbose:
                print("Configuration Schema:")
                print(json.dumps(result["config_schema"], indent=2))
                print()
    except Exception as e:
        if verbose:
            print(f"Config Schema: Not available ({type(e).__name__})")
            print()

    # For RunnableSequence, show the steps
    if isinstance(chain, RunnableSequence):
        steps = getattr(chain, "steps", [])
        if verbose and steps:
            print(f"Sequence Steps ({len(steps)}):")
            for i, step in enumerate(steps, 1):
                step_name = step.get_name() if hasattr(step, "get_name") else type(step).__name__
                print(f"  Step {i}: {step_name}")
            print()

    if verbose:
        print("=" * 80)

    return result


def visualize_chain_graph(
    chain: Runnable[Any, Any],
    config: Optional[RunnableConfig] = None,
    show_details: bool = False
) -> Optional[Graph]:
    """Generate and display graph representation of a chain.

    Creates a visual representation of the chain structure using the get_graph()
    method. This helps visualize data flow through the chain and identify potential
    bottlenecks or issues in complex compositions.

    Args:
        chain: The Runnable chain to visualize. Must support the get_graph() method.
        config: Optional RunnableConfig to pass to get_graph(). Can affect the
            structure of configurable chains.
        show_details: If True, prints detailed information about each node and edge.
            If False, only prints a summary.

    Returns:
        The Graph object if the chain supports graph generation, None otherwise.

    Raises:
        AttributeError: If the chain does not support get_graph() method.
        TypeError: If get_graph() fails due to type issues.

    Example:
        >>> from langchain_core.prompts import ChatPromptTemplate
        >>> from langchain_core.output_parsers import StrOutputParser
        >>> 
        >>> # Create LCEL chain
        >>> prompt = ChatPromptTemplate.from_template("Translate {text} to {language}")
        >>> parser = StrOutputParser()
        >>> chain = prompt | parser
        >>> 
        >>> # Visualize the chain
        >>> graph = visualize_chain_graph(chain, show_details=True)
        >>> if graph:
        >>>     print(f"Chain has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    """
    try:
        graph = chain.get_graph(config=config)

        print("=" * 80)
        print(f"Chain Graph Visualization: {chain.get_name() if hasattr(chain, 'get_name') else type(chain).__name__}")
        print("=" * 80)
        print()

        print(f"Summary: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        print()

        if show_details:
            print("Nodes:")
            print("-" * 80)
            for node_id, node in graph.nodes.items():
                print(f"ID: {node_id}")
                print(f"  Name: {node.name}")
                if node.data:
                    if hasattr(node.data, "__name__"):
                        print(f"  Data Type: {node.data.__name__}")
                    elif hasattr(node.data, "get_name"):
                        print(f"  Data Type: {node.data.get_name()}")
                    else:
                        print(f"  Data Type: {type(node.data).__name__}")
                if node.metadata:
                    print(f"  Metadata: {node.metadata}")
                print()

            print("Edges:")
            print("-" * 80)
            for i, edge in enumerate(graph.edges, 1):
                source_node = graph.nodes.get(edge.source)
                target_node = graph.nodes.get(edge.target)
                print(f"Edge {i}:")
                print(f"  {source_node.name if source_node else edge.source} -> "
                      f"{target_node.name if target_node else edge.target}")
                if edge.conditional:
                    print(f"  Type: Conditional")
                if edge.data:
                    print(f"  Data: {edge.data}")
                print()

        # Print ASCII representation of the graph
        print("Graph Flow:")
        print("-" * 80)
        _print_graph_flow(graph)
        print()
        print("=" * 80)

        return graph

    except AttributeError as e:
        print(f"Error: Chain does not support graph visualization")
        print(f"Details: {type(chain).__name__} does not have get_graph() method")
        print(f"Technical: {str(e)}")
        return None
    except TypeError as e:
        print(f"Error: Failed to generate graph due to type issues")
        print(f"Details: {str(e)}")
        return None
    except Exception as e:
        print(f"Error: Unexpected error during graph visualization")
        print(f"Type: {type(e).__name__}")
        print(f"Details: {str(e)}")
        return None


def _print_graph_flow(graph: Graph) -> None:
    """Print ASCII representation of graph flow.

    Helper function to print a simple ASCII visualization of the graph structure.

    Args:
        graph: The Graph object to visualize.
    """
    # Find first node (no incoming edges)
    first_node = graph.first_node()
    if not first_node:
        print("(No clear starting node)")
        return

    # Track visited nodes to avoid cycles
    visited = set()
    _print_node_flow(graph, first_node, visited, indent=0)


def _print_node_flow(graph: Graph, node: Node, visited: set, indent: int) -> None:
    """Recursively print node flow with indentation.

    Args:
        graph: The Graph object.
        node: Current node to print.
        visited: Set of visited node IDs.
        indent: Current indentation level.
    """
    if node.id in visited:
        print("  " * indent + f"[{node.name}] (already shown)")
        return

    visited.add(node.id)
    print("  " * indent + f"[{node.name}]")

    # Find outgoing edges
    outgoing = [e for e in graph.edges if e.source == node.id]
    for edge in outgoing:
        target_node = graph.nodes.get(edge.target)
        if target_node:
            arrow = " ---> " if not edge.conditional else " -?-> "
            print("  " * indent + arrow)
            _print_node_flow(graph, target_node, visited, indent + 1)


def get_chain_config(
    chain: Runnable[Any, Any],
    include_schema: bool = True
) -> Dict[str, Any]:
    """Extract chain configuration and parameters.

    Retrieves the configuration settings, parameters, and schema information for
    a chain. Useful for understanding how a chain is configured and what options
    are available for customization.

    Args:
        chain: The Runnable chain to extract configuration from.
        include_schema: If True, includes the JSON schema for the configuration.
            If False, only includes basic configuration information.

    Returns:
        A dictionary containing:
            - "name": Chain name
            - "type": Chain type
            - "config_specs": List of configurable field specifications
            - "schema": JSON schema for configuration (if include_schema=True)
            - "input_schema": Input schema information
            - "output_schema": Output schema information

    Example:
        >>> from langchain_core.runnables import RunnableLambda
        >>> 
        >>> chain = RunnableLambda(lambda x: x.upper())
        >>> config = get_chain_config(chain, include_schema=True)
        >>> print(f"Chain: {config['name']}")
        >>> print(f"Configurable fields: {len(config['config_specs'])}")
    """
    config_info: Dict[str, Any] = {
        "name": chain.get_name() if hasattr(chain, "get_name") else type(chain).__name__,
        "type": type(chain).__name__,
        "config_specs": [],
        "schema": None,
        "input_schema": None,
        "output_schema": None,
    }

    # Get configurable fields
    try:
        if hasattr(chain, "config_specs"):
            specs = chain.config_specs
            config_info["config_specs"] = [
                {
                    "id": spec.id,
                    "name": getattr(spec, "name", spec.id),
                    "description": getattr(spec, "description", ""),
                }
                for spec in specs
            ]
    except Exception:
        pass

    # Get config schema
    if include_schema:
        try:
            schema = chain.config_schema()
            if schema:
                config_info["schema"] = schema.model_json_schema()
        except Exception:
            pass

    # Get input/output schemas
    try:
        input_schema = chain.get_input_schema()
        if input_schema:
            config_info["input_schema"] = input_schema.model_json_schema()
    except Exception:
        pass

    try:
        output_schema = chain.get_output_schema()
        if output_schema:
            config_info["output_schema"] = output_schema.model_json_schema()
    except Exception:
        pass

    return config_info


def print_chain_steps(
    chain: Union[Runnable[Any, Any], RunnableSequence],
    show_types: bool = True
) -> None:
    """Display execution steps in LCEL compositions.

    Prints a detailed breakdown of the steps in a chain, particularly useful for
    understanding LCEL (LangChain Expression Language) pipe compositions. Shows
    the type flow through each step of the chain.

    Args:
        chain: The chain to analyze. Works best with RunnableSequence chains
            created using the pipe operator (|).
        show_types: If True, attempts to show input/output types for each step.
            If False, only shows step names.

    Example:
        >>> from langchain_core.prompts import ChatPromptTemplate
        >>> from langchain_core.output_parsers import StrOutputParser
        >>> 
        >>> # Create LCEL chain with pipe operator
        >>> prompt = ChatPromptTemplate.from_template("Say hello to {name}")
        >>> parser = StrOutputParser()
        >>> chain = prompt | parser
        >>> 
        >>> # Print the steps
        >>> print_chain_steps(chain, show_types=True)
    """
    print("=" * 80)
    print("Chain Execution Steps")
    print("=" * 80)
    print()

    # Check if this is a RunnableSequence
    if isinstance(chain, RunnableSequence):
        steps = getattr(chain, "steps", [])
        if not steps:
            print("No steps found in chain")
            return

        print(f"Total Steps: {len(steps)}")
        print()

        for i, step in enumerate(steps, 1):
            step_name = step.get_name() if hasattr(step, "get_name") else type(step).__name__
            print(f"Step {i}: {step_name}")
            print("-" * 80)
            print(f"  Type: {type(step).__module__}.{type(step).__name__}")

            if show_types:
                # Try to get input/output schema
                try:
                    input_schema = step.get_input_schema()
                    if input_schema:
                        schema_name = getattr(input_schema, "__name__", str(input_schema))
                        print(f"  Input Type: {schema_name}")
                except Exception:
                    print(f"  Input Type: <unable to determine>")

                try:
                    output_schema = step.get_output_schema()
                    if output_schema:
                        schema_name = getattr(output_schema, "__name__", str(output_schema))
                        print(f"  Output Type: {schema_name}")
                except Exception:
                    print(f"  Output Type: <unable to determine>")

            print()

        # Show the type flow
        print("Type Flow:")
        print("-" * 80)
        for i, step in enumerate(steps):
            step_name = step.get_name() if hasattr(step, "get_name") else type(step).__name__
            print(f"  {step_name}", end="")
            if i < len(steps) - 1:
                print(" |")
                print("  " + "↓")
            else:
                print()
        print()

    else:
        # Not a sequence, just show basic info
        chain_name = chain.get_name() if hasattr(chain, "get_name") else type(chain).__name__
        print(f"Single Step: {chain_name}")
        print(f"Type: {type(chain).__module__}.{type(chain).__name__}")
        print()

        if show_types:
            try:
                input_schema = chain.get_input_schema()
                output_schema = chain.get_output_schema()
                input_name = getattr(input_schema, "__name__", str(input_schema)) if input_schema else "<unknown>"
                output_name = getattr(output_schema, "__name__", str(output_schema)) if output_schema else "<unknown>"
                print(f"Input Type: {input_name}")
                print(f"Output Type: {output_name}")
            except Exception:
                print("Type information not available")

    print("=" * 80)


# Example usage demonstrations
if __name__ == "__main__":
    print("LangChain Chain Introspection Utility")
    print("=" * 80)
    print()
    print("This utility provides functions to inspect and analyze LangChain chains.")
    print("Note: Examples require langchain-core and optionally langchain-openai.")
    print()

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnableLambda

        print("Example 1: Simple LCEL Chain")
        print("-" * 80)
        
        # Create a simple LCEL chain
        prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}")
        parser = StrOutputParser()
        simple_chain = prompt | parser

        # Inspect the chain
        print("\n>>> inspect_chain(simple_chain)")
        inspect_chain(simple_chain, verbose=True)

        print("\n" * 2)
        print("Example 2: Chain Graph Visualization")
        print("-" * 80)
        
        # Visualize the chain
        print("\n>>> visualize_chain_graph(simple_chain, show_details=True)")
        visualize_chain_graph(simple_chain, show_details=True)

        print("\n" * 2)
        print("Example 3: Chain Configuration")
        print("-" * 80)
        
        # Get chain configuration
        print("\n>>> get_chain_config(simple_chain)")
        config = get_chain_config(simple_chain, include_schema=True)
        print(json.dumps(config, indent=2, default=str))

        print("\n" * 2)
        print("Example 4: Chain Steps")
        print("-" * 80)
        
        # Print chain steps
        print("\n>>> print_chain_steps(simple_chain, show_types=True)")
        print_chain_steps(simple_chain, show_types=True)

        print("\n" * 2)
        print("Example 5: Complex Chain with Lambda")
        print("-" * 80)
        
        # Create a more complex chain
        complex_chain = (
            prompt 
            | RunnableLambda(lambda x: x.to_messages()) 
            | RunnableLambda(lambda msgs: str(msgs[0].content))
        )

        print("\n>>> inspect_chain(complex_chain)")
        inspect_chain(complex_chain, verbose=True)

        print("\n>>> print_chain_steps(complex_chain)")
        print_chain_steps(complex_chain, show_types=True)

    except ImportError as e:
        print(f"Error: Missing required dependencies")
        print(f"Please install: pip install langchain-core")
        print(f"Details: {str(e)}")
    except Exception as e:
        print(f"Error running examples: {type(e).__name__}: {str(e)}")

    print("\n" * 2)
    print("=" * 80)
    print("Chain introspection complete!")
    print("=" * 80)
