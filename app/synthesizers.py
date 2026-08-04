from __future__ import annotations

import typing as t

from ragas.dataset_schema import SingleTurnSample
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.base import SingleHopScenario
from ragas.testset.synthesizers.single_hop.prompts import QueryCondition
from ragas.testset.synthesizers.multi_hop.base import MultiHopScenario
from ragas.testset.synthesizers.multi_hop.prompts import QueryConditions

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


def _prepend_faculty(content: str, faculty_name: str | None) -> str:
    if faculty_name:
        return f"[{faculty_name}]\n\n{content}"
    return content


class CustomSingleHopSpecificQuerySynthesizer(SingleHopSpecificQuerySynthesizer):
    async def _generate_sample(
        self, scenario: SingleHopScenario, callbacks: Callbacks
    ) -> SingleTurnSample:
        page_content = scenario.nodes[0].properties.get("page_content", "")
        faculty_name = scenario.nodes[0].properties.get("faculty_name", None)

        prompt_context = _prepend_faculty(page_content, faculty_name)

        prompt_input = QueryCondition(
            persona=scenario.persona,
            term=scenario.term,
            context=prompt_context,
            query_length=scenario.length.value,
            query_style=scenario.style.value,
            llm_context=self.llm_context,
        )
        response = await self.generate_query_reference_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        return SingleTurnSample(
            user_input=response.query,
            reference=response.answer,
            reference_contexts=[page_content],
            persona_name=getattr(scenario.persona, "name", None),
            query_style=getattr(scenario.style, "name", None),
            query_length=getattr(scenario.length, "name", None),
        )


class CustomMultiHopSpecificQuerySynthesizer(MultiHopSpecificQuerySynthesizer):
    async def _generate_sample(
        self, scenario: MultiHopScenario, callbacks: Callbacks
    ) -> SingleTurnSample:
        if not isinstance(scenario, MultiHopScenario):
            raise TypeError("scenario type should be MultiHopScenario")

        clean_contexts: t.List[str] = []
        prompt_contexts: t.List[str] = []
        for i, node in enumerate(scenario.nodes):
            content = node.properties.get("page_content", "")
            tag = f"<{i + 1}-hop>\n\n"
            tagged = f"{tag}{content}"
            clean_contexts.append(tagged)

            faculty_name = node.properties.get("faculty_name", None)
            prompt_contexts.append(_prepend_faculty(tagged, faculty_name))

        prompt_input = QueryConditions(
            persona=scenario.persona,
            themes=scenario.combinations,
            context=prompt_contexts,
            query_length=scenario.length.value,
            query_style=scenario.style.value,
            llm_context=self.llm_context,
        )
        response = await self.generate_query_reference_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        return SingleTurnSample(
            user_input=response.query,
            reference=response.answer,
            reference_contexts=clean_contexts,
            persona_name=getattr(scenario.persona, "name", None),
            query_style=getattr(scenario.style, "name", None),
            query_length=getattr(scenario.length, "name", None),
        )


class CustomMultiHopAbstractQuerySynthesizer(MultiHopAbstractQuerySynthesizer):
    async def _generate_sample(
        self, scenario: MultiHopScenario, callbacks: Callbacks
    ) -> SingleTurnSample:
        if not isinstance(scenario, MultiHopScenario):
            raise TypeError("scenario type should be MultiHopScenario")

        clean_contexts: t.List[str] = []
        prompt_contexts: t.List[str] = []
        for i, node in enumerate(scenario.nodes):
            content = node.properties.get("page_content", "")
            tag = f"<{i + 1}-hop>\n\n"
            tagged = f"{tag}{content}"
            clean_contexts.append(tagged)

            faculty_name = node.properties.get("faculty_name", None)
            prompt_contexts.append(_prepend_faculty(tagged, faculty_name))

        prompt_input = QueryConditions(
            persona=scenario.persona,
            themes=scenario.combinations,
            context=prompt_contexts,
            query_length=scenario.length.value,
            query_style=scenario.style.value,
            llm_context=self.llm_context,
        )
        response = await self.generate_query_reference_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        return SingleTurnSample(
            user_input=response.query,
            reference=response.answer,
            reference_contexts=clean_contexts,
            persona_name=getattr(scenario.persona, "name", None),
            query_style=getattr(scenario.style, "name", None),
            query_length=getattr(scenario.length, "name", None),
        )
