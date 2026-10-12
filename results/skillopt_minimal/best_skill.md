# ALFWorld operating skill
1. Preserve the task goal throughout the episode.
2. Use `go to` to inspect likely receptacles and objects near them before attempting interaction.
3. For tasks requiring object manipulation (take, use): ensure you are holding or directly adjacent to the target object; if not visible in hand, attempt a take action first rather than examining while empty-handed.
4. For look-under-light tasks: go to the lamp location, switch on if necessary, locate the hidden target, and only then examine it once acquired or confirmed present under light.
5. Avoid repeating an action when the observation did not improve; break loops by re-evaluating object state before retrying same interaction type.
6. Select only actions copied verbatim from the admissible-action list to ensure valid syntax.