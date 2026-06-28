# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. User requirements

### 5.1 Plans
You must write implementation plans in the `memory/plans/yyyymmdd-hhmmss-<your plan>.md` when
- move to a new stage
- some additional changes, implementations, fixes that are required in this stage/plan and are not included in the existing tasks, write a sub plan
- significant changes / fixes are needed
- asked by user

In your plan, you must include
```md
# <Name of this plan>

## 0. Reference

<reference link(s) to related doc / plan / website, and the relation of this plan>

## 1. Target

<bref of your target or task>

<success scope or acceptance requirements of this task>

## 2. Scope

<in scope>

<out of scope>

## 3. <task/plan/target...>

<write your plan here>

## 4. Test

<how do you test this implementation>

```

### 5.2 Progress
You must progress in the `memory/progress/yyyymmdd-hhmmss-<your plan>.md` when you create a plan or sub plan .

In this progress, you must include
```md
# <Name of this plan>

## 0. Reference

<reference link(s) to related doc / plan / website, and the relation of this plan>

## 1. Task list
| Task | Status | Acceptance |
| --- | --- | --- |
| ... | ... | ... |

## 2. Memo
...

```

### 5.3 Plan References
After writing plan and progress, you write the references in `memory/references.md` that relate to this plan. Note that you should always append the latest plan from the top. Note that artifact link is `path/to/python_script.py`, don't include the whole span, what you need to write is the **first line of the symbol** that you edited / created in this implementation.
```md
# Plan references

## <name of plan>: yyyy-mm-dd hh:mm:ss

**bref**:
  <write your bref of this plan>

**reference**:
  - <relative path to plan>
  - <relative path to progress>
  - artifacts:
    - <relative path to artifact>; [`<symbole name>`, `<symbole name>`, ...]
    - ...

---

...
```

### 5.4 Simple workflow
Following is a simple workflow
1. Read `memory/reflections.md` and use `karpathy-guidelines` skill
2. Create a plan in `memory/plans`
3. Create a progress in `memory/progress`
4. Ask user to review the plan and progress. If the user doesn't satisfy, revise them.
5. Append reference
6. Implement and test
7. Check the progress and mark the finished tasks and find the remained works. If the tasks are not all fully finished, go back to step 5 and continune to implement the remained tasks.
8.  Final check / test

### 5.5 Reflection

You need to write the reflection in `memory/reflections.md` when the plan / task failed. Note that you should seperate the generic and project specific issue / problem. You should keep the reflection in brief. `Occurrence` means the times that you blundered. Once you blunder or corrected by user, you must find which `<issue or problem>` have you made, and increase the `Occurrence` by 1. If the `<issue or problem>` is new, append it in `Generic` if the scope is generic issue, in `Project` if the scope is project specific issue. DO NOT create the duplicated kind of `<issue or problem>`.
```md
# Reflections
<may includes some user-provided lessons or requirements>

## Generic

### <Issue or problem>
**Occurrence**: N
- Reason: ...
- Cause to: ...
- Lesson: ...

---

## Project

### <Issue or problem>
**Occurrence**: N
- Reason: ...
- Cause to: ...
- Lesson: ...

```


#### 5.5.1 Non-technical failures
You may be asked to revise the implementation / plan after finishing your tasks, that is always because
- You misunderstand or violate the requirement
- You implement some features implicitly that the user did not require
- ...

This is treated as failure even all tests are passed. When the user ask you to discard the plan and reimplement, you should mark the related files as following, as well as the the reference in `memory/references.md`. You must includ the reason of discard and the refer to the new plan/progress. Format is following
```md
# <Name of this plan>
> Status: Rejected and deprecated
>
> Reason for deprecation:
> 1. ...
> ...
>
> Future implementation must switch to the new plan: memory/plans/<...>.md.
```

#### 5.5.2 Technical failures
Technical failures means although you have tested via pytest or some toy test cases, the real case testing failed. That's always because your test cases are too simple or you just use some mock / fake test implemetation, which don't reflect to the real world usage. When the real usage testing failed, you also must reflect to the reason, not only "you didn't use the real world test case", but also include the underlying reason.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.