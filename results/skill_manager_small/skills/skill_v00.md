# ALFWorld operating skill

1. Preserve the task goal throughout the episode.
2. Use `go to` to inspect likely receptacles before trying to take an object.
3. For look-under-light tasks: find and take the target, go to the named lamp,
   switch the lamp on if necessary, then examine the target while holding it.
4. For clean/heat/cool tasks: acquire the object, use the required appliance,
   then place it in the requested destination.
5. Avoid repeating an action when the observation did not improve.
6. Select the numeric ID of one action from the numbered admissible-action list.
