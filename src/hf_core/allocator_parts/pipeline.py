from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import AllocatorContext


@dataclass
class AllocatorPipeline:
    steps: list = field(default_factory=list)

    def run(self, ctx: AllocatorContext) -> AllocatorContext:
        for step in self.steps:
            ctx = step(ctx)
        return ctx
