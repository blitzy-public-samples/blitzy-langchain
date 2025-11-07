"""Chain pipeline where the outputs of one step feed directly into next."""

from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.utils.input import get_color_mapping
from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from langchain_classic.chains.base import Chain


class SequentialChain(Chain):
    """Chain where the outputs of one chain feed directly into next.
    
    SequentialChain executes multiple chains in sequence, passing outputs from
    each chain as inputs to subsequent chains. This enables complex multi-step
    workflows where each step builds on previous results.
    
    **Type Flow Documentation:**
    
    Chain execution follows this pattern:
    1. Initial inputs dict is copied to known_values: Dict[str, Any]
    2. For each chain in sequence:
       - Chain N executes with known_values as input
       - Chain N produces output: Dict[output_key_N: value_N]
       - Output is merged into known_values via dict.update()
       - Chain N+1 can now access output_key_N in its input_keys
    3. Final return filters known_values to only requested output_variables
    
    **Input/Output Key Mapping:**
    
    For chains to connect properly:
    - Chain N's output_keys become available to Chain N+1's input_keys
    - Each chain[i].output_keys must not overlap with existing known_variables
    - Each chain[i+1].input_keys must be satisfied by:
      * Initial input_variables, OR
      * output_keys from any previous chain[0..i], OR
      * memory.memory_variables (if memory is configured)
    
    **Common Pitfall:**
    
    ValueError will be raised if chain[i+1] expects an input key that's not in:
    - input_variables (provided at chain initialization)
    - output_keys from any previous chain
    - memory variables
    
    Solution: Ensure chain[i].output_keys contains all keys needed by 
    chain[i+1].input_keys, or include missing keys in input_variables.
    
    **Debugging Tip:**
    
    Use return_all=True to access all intermediate outputs from the chain
    execution, not just the final output_variables. This helps diagnose
    which chain produced which outputs.
    
    Examples:
        Multi-step document processing with explicit key mapping:
        
        ```python
        from langchain_classic.chains import LLMChain, SequentialChain
        from langchain_classic.prompts import PromptTemplate
        from langchain_openai import OpenAI
        
        llm = OpenAI(temperature=0.7)
        
        # Chain 1: Generate outline from topic
        outline_template = PromptTemplate(
            input_variables=["topic"],
            template="Create an outline for an article about {topic}"
        )
        outline_chain = LLMChain(
            llm=llm,
            prompt=outline_template,
            output_key="outline"  # This becomes available to next chain
        )
        
        # Chain 2: Expand outline into draft
        draft_template = PromptTemplate(
            input_variables=["outline", "topic"],  # Can access previous output + original input
            template="Write a draft article based on this outline:\\n{outline}\\n\\nTopic: {topic}"
        )
        draft_chain = LLMChain(
            llm=llm,
            prompt=draft_template,
            output_key="draft"
        )
        
        # Chain 3: Polish the draft
        polish_template = PromptTemplate(
            input_variables=["draft"],
            template="Polish and improve this draft:\\n{draft}"
        )
        polish_chain = LLMChain(
            llm=llm,
            prompt=polish_template,
            output_key="final_article"
        )
        
        # Compose sequential chain
        sequential_chain = SequentialChain(
            chains=[outline_chain, draft_chain, polish_chain],
            input_variables=["topic"],  # Initial inputs
            output_variables=["final_article"],  # What to return
            verbose=True
        )
        
        # Execute: topic → outline → draft → final_article
        result = sequential_chain({"topic": "artificial intelligence"})
        print(result["final_article"])
        
        # Get all intermediate outputs
        sequential_chain_all = SequentialChain(
            chains=[outline_chain, draft_chain, polish_chain],
            input_variables=["topic"],
            return_all=True  # Returns all outputs, not just final
        )
        result_all = sequential_chain_all({"topic": "artificial intelligence"})
        # result_all contains: {"outline": ..., "draft": ..., "final_article": ...}
        ```
        
        Common error scenario and solution:
        
        ```python
        # ERROR: Variable name mismatch
        chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="summary")
        chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="analysis")
        # chain2 expects input_variables=["summary_text"] but chain1 outputs "summary"
        
        # This will raise: ValueError: Missing required input keys: {'summary_text'}
        # Solution: Make chain1.output_key match chain2's expected input
        chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="summary_text")
        ```
    
    Source: libs/langchain/langchain_classic/chains/sequential.py:16-43
    """

    chains: list[Chain]
    input_variables: list[str]
    output_variables: list[str]  #: :meta private:
    return_all: bool = False

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def input_keys(self) -> list[str]:
        """Return expected input keys to the chain.

        :meta private:
        """
        return self.input_variables

    @property
    def output_keys(self) -> list[str]:
        """Return output key.

        :meta private:
        """
        return self.output_variables

    @model_validator(mode="before")
    @classmethod
    def validate_chains(cls, values: dict) -> Any:
        """Validate that the correct inputs exist for all chains.
        
        Performs comprehensive validation to ensure chains can be composed:
        1. Verifies no overlap between input_variables and memory_variables
        2. Validates each chain's input_keys are satisfied by known variables
        3. Ensures no chain produces output_keys that already exist
        4. Confirms output_variables are produced by the chain sequence
        
        Args:
            values: Dictionary containing chain configuration with keys:
                - chains: List of Chain objects to execute sequentially
                - input_variables: List of input keys provided at execution time
                - memory: Optional BaseMemory instance for conversation context
                - output_variables: Optional list of keys to return (auto-computed if not provided)
                - return_all: Optional bool to return all intermediate outputs
        
        Returns:
            Updated values dict with auto-computed output_variables if not provided.
            If return_all=True and output_variables not specified, returns all outputs
            except input_variables. Otherwise, returns last chain's output_keys.
        
        Raises:
            ValueError: In the following scenarios:
                - Input variables overlap with memory keys (would cause ambiguity)
                - Chain requires input key not in known_variables (missing dependency)
                - Chain produces output key that already exists (duplicate key)
                - Requested output_variables are not produced by any chain
        
        Common ValidationError Scenarios:
            1. Missing required input keys:
               - Cause: chain[i+1].input_keys contains key not in prior outputs
               - Solution: Add key to input_variables or ensure prior chain outputs it
            
            2. Overlapping memory and input keys:
               - Cause: Same key name in both input_variables and memory.memory_variables
               - Solution: Rename variables to avoid collision
            
            3. Duplicate output keys:
               - Cause: Multiple chains produce same output_key
               - Solution: Use unique output_key for each chain
            
            4. Expected output not found:
               - Cause: output_variables contains key not produced by any chain
               - Solution: Verify chain output_keys match expected variables
        
        Source: libs/langchain/langchain_classic/chains/sequential.py:45-96
        """
        chains = values["chains"]
        input_variables = values["input_variables"]
        memory_keys = []
        if "memory" in values and values["memory"] is not None:
            """Validate that prompt input variables are consistent."""
            memory_keys = values["memory"].memory_variables
            if set(input_variables).intersection(set(memory_keys)):
                overlapping_keys = set(input_variables) & set(memory_keys)
                msg = (
                    f"The input key(s) {''.join(overlapping_keys)} are found "
                    f"in the Memory keys ({memory_keys}) - please use input and "
                    f"memory keys that don't overlap."
                )
                raise ValueError(msg)

        known_variables = set(input_variables + memory_keys)

        for chain in chains:
            missing_vars = set(chain.input_keys).difference(known_variables)
            if chain.memory:
                missing_vars = missing_vars.difference(chain.memory.memory_variables)

            if missing_vars:
                msg = (
                    f"Missing required input keys: {missing_vars}, "
                    f"only had {known_variables}"
                )
                raise ValueError(msg)
            overlapping_keys = known_variables.intersection(chain.output_keys)
            if overlapping_keys:
                msg = f"Chain returned keys that already exist: {overlapping_keys}"
                raise ValueError(msg)

            known_variables |= set(chain.output_keys)

        if "output_variables" not in values:
            if values.get("return_all", False):
                output_keys = known_variables.difference(input_variables)
            else:
                output_keys = chains[-1].output_keys
            values["output_variables"] = output_keys
        else:
            missing_vars = set(values["output_variables"]).difference(known_variables)
            if missing_vars:
                msg = f"Expected output variables that were not found: {missing_vars}."
                raise ValueError(msg)

        return values

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        """Execute all chains in sequence, accumulating outputs.
        
        Type flow: Each chain receives the accumulated known_values dict and adds
        its outputs, making them available to subsequent chains.
        
        Args:
            inputs: Initial input dictionary containing all keys specified in
                self.input_variables. Must be a dict mapping variable names to
                their string values (e.g., {"topic": "AI", "style": "formal"}).
            run_manager: Optional callback manager for monitoring chain execution.
                If None, a no-op manager is used. Callbacks receive events for
                each chain invocation via get_child().
        
        Returns:
            Dictionary containing only the requested output_variables. Keys are
            filtered from the final known_values dict. Structure matches:
            {output_var_1: value_1, output_var_2: value_2, ...}
        
        Raises:
            KeyError: If a chain expects an input key that's not in known_values.
                This indicates a validation error that should have been caught
                during chain initialization.
            Exception: Any exception raised by individual chain execution is
                propagated. Check chain-specific documentation for possible errors.
        
        Type Flow Visualization:
            known_values starts as: {"topic": "AI"}
            After chain[0]: {"topic": "AI", "outline": "..."}
            After chain[1]: {"topic": "AI", "outline": "...", "draft": "..."}
            After chain[2]: {"topic": "AI", "outline": "...", "draft": "...", "final": "..."}
            Return filters to: {"final": "..."}  # Only output_variables
        
        Source: libs/langchain/langchain_classic/chains/sequential.py:98-109
        """
        # Type Flow: Start with initial inputs, will accumulate all chain outputs
        known_values = inputs.copy()
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        for _i, chain in enumerate(self.chains):
            callbacks = _run_manager.get_child()
            # Type Flow: chain() returns Dict with keys from chain.output_keys
            # Example: chain.output_keys=["summary"] → outputs={"summary": "..."}
            outputs = chain(known_values, return_only_outputs=True, callbacks=callbacks)
            # Type Flow: dict.update() merges chain outputs into known_values,
            # making them available to next chain's input_keys
            # Example: known_values={"topic": "AI"} + outputs={"summary": "..."} 
            #          → known_values={"topic": "AI", "summary": "..."}
            known_values.update(outputs)
        # Type Flow: Final return filters known_values to only requested output_variables
        # This allows chains to produce intermediate outputs not exposed to caller
        return {k: known_values[k] for k in self.output_variables}

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Execute all chains asynchronously in sequence, accumulating outputs.
        
        Async version of _call() with identical type flow and behavior. Chains are
        executed sequentially (not in parallel) using await, preserving output
        dependencies between chains.
        
        Args:
            inputs: Initial input dictionary containing all keys specified in
                self.input_variables. Can contain any JSON-serializable values.
            run_manager: Optional async callback manager for monitoring chain
                execution. If None, a no-op async manager is used. Async callbacks
                must not block the event loop.
        
        Returns:
            Dictionary containing only the requested output_variables. Keys are
            filtered from the final known_values dict. Structure matches:
            {output_var_1: value_1, output_var_2: value_2, ...}
        
        Raises:
            KeyError: If a chain expects an input key that's not in known_values.
            Exception: Any exception raised by individual async chain execution.
        
        Async Execution Notes:
            - Chains are executed sequentially, not in parallel, because each chain
              depends on outputs from previous chains
            - Use this method when chains involve async I/O operations (API calls,
              async database queries, etc.)
            - Ensure all chains support async execution via acall()
            - Async callbacks must be non-blocking and properly await coroutines
        
        Type Flow Visualization:
            Same as _call(), but with async execution:
            known_values starts as: {"query": "..."}
            After await chain[0].acall(): {"query": "...", "context": "..."}
            After await chain[1].acall(): {"query": "...", "context": "...", "answer": "..."}
            Return: {"answer": "..."}  # Only output_variables
        
        Source: libs/langchain/langchain_classic/chains/sequential.py:111-126
        """
        # Type Flow: Start with initial inputs, will accumulate all chain outputs
        known_values = inputs.copy()
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        for _i, chain in enumerate(self.chains):
            # Type Flow: await chain.acall() returns Dict with keys from chain.output_keys
            # Each chain executes asynchronously but sequentially (not parallel)
            outputs = await chain.acall(
                known_values,
                return_only_outputs=True,
                callbacks=callbacks,
            )
            # Type Flow: dict.update() merges chain outputs into known_values,
            # making them available to next chain's input_keys
            known_values.update(outputs)
        # Type Flow: Final return filters known_values to only requested output_variables
        return {k: known_values[k] for k in self.output_variables}


class SimpleSequentialChain(Chain):
    """Simple chain where the outputs of one step feed directly into next.
    
    SimpleSequentialChain is a specialized version of SequentialChain optimized for
    the common case where each chain has exactly one input and one output, with
    outputs flowing directly as strings from one chain to the next.
    
    **Simplified Type Flow:**
    
    Unlike SequentialChain which uses dict key mapping, SimpleSequentialChain passes
    raw string values:
    1. Initial input: {"input": str_value}
    2. chain[0].run(str_value) → str_output_0
    3. chain[1].run(str_output_0) → str_output_1
    4. chain[N].run(str_output_N-1) → str_output_N
    5. Final output: {"output": str_output_N}
    
    **Constraint:**
    
    All chains MUST have exactly one input_key and one output_key. This is validated
    during initialization and will raise ValueError if violated.
    
    **Difference from SequentialChain:**
    
    - SimpleSequentialChain: Direct string-to-string passing, no key mapping needed
    - SequentialChain: Dict-based with explicit key mapping between chains
    
    Use SimpleSequentialChain when:
    - All chains have single input/output
    - You want simpler chain composition without managing key names
    - Output of chain N is exactly the input needed for chain N+1
    
    Use SequentialChain when:
    - Chains have multiple inputs/outputs
    - You need to preserve multiple intermediate values
    - Chains require specific input key names
    
    Examples:
        Simple 3-step text processing pipeline:
        
        ```python
        from langchain_classic.chains import LLMChain, SimpleSequentialChain
        from langchain_classic.prompts import PromptTemplate
        from langchain_openai import OpenAI
        
        llm = OpenAI(temperature=0.7)
        
        # Each chain has one input, one output
        summarize = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["text"],
                template="Summarize this text:\\n{text}"
            )
        )
        
        translate = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["text"],
                template="Translate to Spanish:\\n{text}"
            )
        )
        
        polish = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["text"],
                template="Make this more formal:\\n{text}"
            )
        )
        
        # String flows: input → summarize → translate → polish → output
        simple_chain = SimpleSequentialChain(
            chains=[summarize, translate, polish],
            verbose=True,
            strip_outputs=True  # Remove extra whitespace between steps
        )
        
        result = simple_chain.run("Long article text here...")
        print(result)  # Final polished Spanish summary
        ```
    
    Source: libs/langchain/langchain_classic/chains/sequential.py:129-156
    """

    chains: list[Chain]
    strip_outputs: bool = False
    input_key: str = "input"  #: :meta private:
    output_key: str = "output"  #: :meta private:

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def input_keys(self) -> list[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Return output key.

        :meta private:
        """
        return [self.output_key]

    @model_validator(mode="after")
    def validate_chains(self) -> Self:
        """Validate that chains are all single input/output.
        
        Ensures all chains meet SimpleSequentialChain's requirement of exactly
        one input_key and one output_key, enabling direct string-to-string passing.
        
        Args:
            self: The SimpleSequentialChain instance being validated (after init).
        
        Returns:
            Self: The validated chain instance, unchanged.
        
        Raises:
            ValueError: If any chain has multiple input keys (len(input_keys) != 1).
                Error message includes the problematic chain and its input count.
            ValueError: If any chain has multiple output keys (len(output_keys) != 1).
                Error message includes the problematic chain and its output count.
        
        Common Validation Errors:
            1. "Chains used in SimplePipeline should all have one input":
               - Cause: Chain requires multiple inputs (e.g., template with multiple variables)
               - Solution: Use SequentialChain instead, or modify chain to accept single input
            
            2. "Chains used in SimplePipeline should all have one output":
               - Cause: Chain produces multiple outputs (e.g., multiple output_keys)
               - Solution: Use SequentialChain instead, or modify chain to produce single output
        
        Source: libs/langchain/langchain_classic/chains/sequential.py:158-174
        """
        for chain in self.chains:
            if len(chain.input_keys) != 1:
                msg = (
                    "Chains used in SimplePipeline should all have one input, got "
                    f"{chain} with {len(chain.input_keys)} inputs."
                )
                raise ValueError(msg)
            if len(chain.output_keys) != 1:
                msg = (
                    "Chains used in SimplePipeline should all have one output, got "
                    f"{chain} with {len(chain.output_keys)} outputs."
                )
                raise ValueError(msg)
        return self

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        """Execute chains sequentially with direct string-to-string passing.
        
        Type flow: String value flows through each chain via chain.run(), with
        each chain's output becoming the next chain's input.
        
        Args:
            inputs: Input dictionary with single key matching self.input_key.
                Structure: {self.input_key: str_value}
                Example: {"input": "Process this text"}
            run_manager: Optional callback manager for monitoring execution.
                Each chain step receives a child callback manager tagged with
                step number (step_1, step_2, etc.) for granular tracking.
        
        Returns:
            Dictionary with single key matching self.output_key containing final
            string output from last chain. Structure: {self.output_key: final_str}
        
        Raises:
            KeyError: If inputs dict doesn't contain self.input_key.
            Exception: Any exception raised by individual chain.run() execution.
        
        Type Flow Visualization:
            inputs: {"input": "Original text"}
            After chain[0].run(): _input = "Summarized text"
            After chain[1].run(): _input = "Translated text"
            After chain[2].run(): _input = "Polished text"
            Return: {"output": "Polished text"}
        
        String Transformation:
            - Each chain.run() takes str input, returns str output
            - If strip_outputs=True, whitespace is removed between steps
            - Intermediate outputs are printed to callbacks for monitoring
            - Final output is wrapped in dict with self.output_key
        
        Source: libs/langchain/langchain_classic/chains/sequential.py:176-197
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        # Type Flow: Extract string value from input dict
        _input = inputs[self.input_key]
        color_mapping = get_color_mapping([str(i) for i in range(len(self.chains))])
        for i, chain in enumerate(self.chains):
            # Type Flow: chain.run(str) → str, direct string transformation
            # Each chain's output becomes next chain's input
            _input = chain.run(
                _input,
                callbacks=_run_manager.get_child(f"step_{i + 1}"),
            )
            # Optional: Strip whitespace from output before passing to next chain
            if self.strip_outputs:
                _input = _input.strip()
            # Log intermediate output for monitoring
            _run_manager.on_text(
                _input,
                color=color_mapping[str(i)],
                end="\n",
                verbose=self.verbose,
            )
        # Type Flow: Wrap final string output in dict with self.output_key
        return {self.output_key: _input}

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Execute chains asynchronously with direct string-to-string passing.
        
        Async version of _call() with identical type flow and behavior. Chains are
        executed sequentially (not in parallel) using await, with string outputs
        flowing directly from one chain to the next.
        
        Args:
            inputs: Input dictionary with single key matching self.input_key.
                Structure: {self.input_key: any_value}
                Value can be any type that chain.arun() accepts.
            run_manager: Optional async callback manager for monitoring execution.
                Each chain step receives a child async callback tagged with step
                number (step_1, step_2, etc.).
        
        Returns:
            Dictionary with single key matching self.output_key containing final
            output from last chain. Structure: {self.output_key: final_value}
        
        Raises:
            KeyError: If inputs dict doesn't contain self.input_key.
            Exception: Any exception raised by individual chain.arun() execution.
        
        Async Execution Notes:
            - Chains execute sequentially (not parallel) since each depends on
              previous output
            - Use when chains involve async I/O operations (API calls, async LLMs)
            - Ensure all chains support async execution via arun()
            - Async callbacks (on_text) must be properly awaited
        
        Type Flow Visualization:
            Same as _call(), but with async execution:
            inputs: {"input": "Original text"}
            After await chain[0].arun(): _input = "Processed by chain 0"
            After await chain[1].arun(): _input = "Processed by chain 1"
            Return: {"output": "Processed by chain 1"}
        
        Source: libs/langchain/langchain_classic/chains/sequential.py:199-220
        """
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        # Type Flow: Extract value from input dict
        _input = inputs[self.input_key]
        color_mapping = get_color_mapping([str(i) for i in range(len(self.chains))])
        for i, chain in enumerate(self.chains):
            # Type Flow: await chain.arun(value) → value, async transformation
            # Each chain's output becomes next chain's input
            _input = await chain.arun(
                _input,
                callbacks=_run_manager.get_child(f"step_{i + 1}"),
            )
            # Optional: Strip whitespace from output before passing to next chain
            if self.strip_outputs:
                _input = _input.strip()
            # Async: await callback for logging intermediate output
            await _run_manager.on_text(
                _input,
                color=color_mapping[str(i)],
                end="\n",
                verbose=self.verbose,
            )
        # Type Flow: Wrap final output in dict with self.output_key
        return {self.output_key: _input}
