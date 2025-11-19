# Whiteboarding Practice Guide

## Introduction

Whiteboarding—whether physical or virtual—is a critical interview skill. It's not just about drawing; it's about communicating complex technical ideas visually and structuring your thinking in real-time.

**Common Scenarios**:
- System architecture design
- Algorithm explanation
- Data flow diagrams
- Performance analysis
- Debugging scenarios

---

## Why Whiteboarding Matters

### Benefits
- **Clarifies thinking**: Forces you to structure ideas
- **Enables collaboration**: Interviewer can follow your thought process
- **Demonstrates communication**: Shows how you explain to others
- **Reveals problem-solving**: Shows systematic approach

### What Interviewers Look For
- **Clarity**: Can they understand your diagram?
- **Structure**: Is it organized logically?
- **Completeness**: Did you cover all components?
- **Interactivity**: Do you engage with the interviewer?
- **Adaptability**: Can you modify based on feedback?

---

## Whiteboarding Fundamentals

### Visual Vocabulary

**Boxes (Rectangles)**:
```
┌──────────────┐
│  Component   │  = Service, system, database
└──────────────┘
```

**Arrows**:
```
A ──→ B   = Flow, dependency, communication
A ←──→ B  = Bidirectional communication
A ···→ B  = Async/optional flow
```

**Cylinders (Databases)**:
```
  ╭─────╮
  │ DB  │  = Database, persistent storage
  ╰─────╯
```

**Clouds (External Systems)**:
```
   ☁
 / │ \   = External service, cloud provider
```

**Queues**:
```
[→ → →]  = Message queue, buffer
```

**Decision Points**:
```
    ◇
   / \   = Conditional logic, decision point
```

### Layout Principles

**Top-Down Flow**:
```
Client
  ↓
API Layer
  ↓
Business Logic
  ↓
Data Layer
```

**Left-to-Right Flow**:
```
Request → Gateway → Service → Database
                      ↓
                   Response
```

**Layered Architecture**:
```
┌────────────────────────────────┐
│    Presentation Layer          │
├────────────────────────────────┤
│    Business Logic              │
├────────────────────────────────┤
│    Data Access                 │
└────────────────────────────────┘
```

---

## System Design Whiteboarding

### Template: LLM Serving System

**Step 1: Start with User (1 min)**
```
 ┌─────┐
 │User │
 └─────┘
   ↓
   ?
```

Say: "Let me start with the user making a request and work through the system."

**Step 2: Add Entry Point (1 min)**
```
 ┌─────┐
 │User │
 └─────┘
   ↓
┌──────────────┐
│Load Balancer │
└──────────────┘
   ↓
   ?
```

**Step 3: Build Out Core Components (3-5 min)**
```
 ┌─────┐
 │User │
 └─────┘
   ↓
┌──────────────┐
│Load Balancer │
└──────────────┘
   ↓
┌──────────────┐
│ API Gateway  │
└──────────────┘
   ↓
┌──────────────┐
│Request Queue │ ← (Redis/Kafka)
└──────────────┘
   ↓
┌──────────────┐
│  Scheduler   │
└──────────────┘
   ↓
┌──────────────┐
│ vLLM Engine  │
│  (8x GPUs)   │
└──────────────┘
   ↓
 Response
```

**Step 4: Add Supporting Systems (2-3 min)**
```
                    ┌────────────┐
                    │Monitoring  │
                    └────────────┘
                         ↑
 ┌─────┐                 │
 │User │           (metrics)
 └─────┘                 │
   ↓                     │
┌──────────────┐         │
│Load Balancer │─────────┤
└──────────────┘         │
   ↓                     │
┌──────────────┐         │
│ API Gateway  │─────────┤
└──────────────┘         │
   ↓                     │
┌──────────────┐         │
│Request Queue │─────────┤
└──────────────┘         │
   ↓                     │
┌──────────────┐         │
│  Scheduler   │─────────┤
└──────────────┘         │
   ↓                     │
┌──────────────┐         │
│ vLLM Engine  │─────────┘
│  (8x GPUs)   │
└──────────────┘
       ↓
   ┌────────┐
   │KV Cache│ (shared storage)
   └────────┘
```

**Step 5: Annotate with Details (2-3 min)**
```
Add labels:
- Load Balancer: "Layer 7, health checks"
- API Gateway: "Auth, rate limiting"
- Queue: "Priority queues (3 levels)"
- Scheduler: "Continuous batching, PagedAttention"
- vLLM: "Tensor Parallel=8, batch_size=100-150"
```

### Whiteboarding Tips for System Design

**Do**:
✓ Start simple, add complexity incrementally
✓ Label components clearly
✓ Use consistent shapes and arrows
✓ Add annotations for key details
✓ Ask "Should I go deeper on X?" before drilling down
✓ Leave space for additions (don't fill entire board)

**Don't**:
❌ Start with all details at once (overwhelming)
❌ Draw tiny, cramped diagrams
❌ Use ambiguous shapes or labels
❌ Erase excessively (shows confusion)
❌ Draw without explaining (talk through it!)

---

## Algorithm Whiteboarding

### Example: Explaining PagedAttention

**Step 1: Show the Problem (1 min)**
```
Traditional Attention Memory:

Sequence 1: [■■■■■■■■■■■□□□□□] ← allocated 100, using 70
Sequence 2: [■■■■■■■□□□□□□□□] ← allocated 100, using 50
Sequence 3: [■■■■■■■■■■□□□□□] ← allocated 100, using 60

Wasted: 30+50+40 = 120 units (40% waste!)
Can't fit Sequence 4 (needs 100) despite 120 free!
```

Say: "The problem is fragmentation. Each sequence allocates contiguous memory, but doesn't use it all."

**Step 2: Show the Solution (2 min)**
```
PagedAttention with Blocks (block_size=16):

Physical Memory (blocks):
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │
└────┴────┴────┴────┴────┴────┴────┴────┘

Sequence 1 (70 tokens = 5 blocks):
Block Table: [3, 1, 7, 4, 2]
             ↓  ↓  ↓  ↓  ↓
        Physical blocks (non-contiguous!)

Sequence 2 (50 tokens = 4 blocks):
Block Table: [5, 0, 6, ...]

No fragmentation! Blocks allocated on-demand.
```

**Step 3: Show the Algorithm (2-3 min)**
```
Attention Computation:

For each query position i:
  1. Compute which block: block_num = i / 16
  2. Lookup physical block: phys = block_table[block_num]
  3. Compute offset: offset = i % 16
  4. Access K,V: kv_cache[phys][offset]

Example: Token 35 in sequence
  block_num = 35 / 16 = 2
  phys = block_table[2] = 7
  offset = 35 % 16 = 3
  → Access kv_cache[7][3]
```

**Step 4: Show the Benefit (30 sec)**
```
Memory Utilization:

Traditional: 60% (due to fragmentation)
PagedAttention: 95% (minimal waste)

Result: 2-5x more concurrent sequences!
```

### Whiteboarding Tips for Algorithms

**Do**:
✓ Start with intuition/problem
✓ Use concrete examples (not just abstract)
✓ Show step-by-step execution
✓ Annotate with explanations
✓ Compare to baseline/naive approach

**Don't**:
❌ Jump to code without visual explanation
❌ Use variables without defining them
❌ Skip the "why" (just show "what")
❌ Assume interviewer follows without checking

---

## Data Flow Whiteboarding

### Example: Request Lifecycle in vLLM

**Step-by-Step Flow**:

```
1. Request Arrival
   ┌──────┐
   │Client│
   └───┬──┘
       │ POST /generate
       ↓
   ┌──────────┐
   │API Server│
   └────┬─────┘
        │

2. Validation & Queueing
        │
        ↓ validate, assign priority
   ┌─────────────┐
   │Request Queue│
   └─────┬───────┘
         │ (waiting for GPU)

3. Scheduling
         ↓
   ┌──────────┐
   │Scheduler │ ← checks GPU memory
   └────┬─────┘
        │ if fits, allocate blocks
        ↓
   ┌─────────────┐
   │Block Manager│ (allocates KV cache blocks)
   └─────┬───────┘
        │

4. Execution
        ↓
   ┌─────────────┐
   │   Worker    │
   │  (GPU)      │
   │  - Prefill  │ → Fills KV cache
   │  - Decode   │ → Generates tokens
   └─────┬───────┘
        │

5. Completion
        ↓ (when done)
   ┌─────────────┐
   │Block Manager│ (frees blocks)
   └─────┬───────┘
        │
        ↓
   ┌──────────┐
   │API Server│ (returns response)
   └────┬─────┘
        │
        ↓
   ┌──────┐
   │Client│
   └──────┘
```

**Narration**:
1. "Request comes in from client via REST API"
2. "API server validates and puts in priority queue"
3. "Scheduler checks if GPU has memory, asks block manager to allocate"
4. "Worker executes: first prefill, then decode loop, using allocated blocks"
5. "On completion, blocks freed, response returned to client"

---

## Virtual Whiteboarding Tools

### Recommended Tools

**For Interviews**:
1. **Excalidraw** (excalidraw.com)
   - Simple, fast
   - Hand-drawn aesthetic
   - Real-time collaboration
   - Free

2. **Miro** (miro.com)
   - Professional
   - Templates available
   - Good for complex diagrams
   - Free tier available

3. **draw.io** (app.diagrams.net)
   - Powerful, feature-rich
   - Professional diagrams
   - Free

4. **Google Jamboard**
   - Simple, built into Google
   - Touch/pen support
   - Easy sharing

### Virtual Whiteboarding Tips

**Setup**:
- Test tool beforehand (practice session)
- Have second monitor (see interviewer + board)
- Use stylus/pen if available (better than mouse)
- Ensure screen share works

**During Interview**:
- Ask "Can you see my screen clearly?"
- Narrate as you draw
- Check in frequently ("Does this make sense?")
- Use colors for emphasis (but not too many)

---

## Practice Exercises

### Exercise 1: System Design Whiteboard (30 min)

**Task**: Design a high-throughput LLM serving system on whiteboard

**Process**:
1. Draw on physical whiteboard or virtual tool
2. Time yourself: Complete in 30 minutes
3. Record yourself explaining
4. Watch back and critique

**Evaluation**:
- [ ] Clear component labels
- [ ] Logical flow (top-down or left-right)
- [ ] Key details annotated
- [ ] Explained while drawing
- [ ] Answered hypothetical follow-ups

### Exercise 2: Algorithm Explanation (15 min)

**Task**: Explain FlashAttention algorithm visually

**Elements to Include**:
- Problem (standard attention memory)
- Tiling approach
- Step-by-step computation
- Memory savings

### Exercise 3: Request Flow (15 min)

**Task**: Draw request lifecycle in vLLM

**Requirements**:
- All key components
- State transitions
- Data flow arrows
- Annotations

### Exercise 4: Debugging Scenario (20 min)

**Task**: Given "P95 latency is 5x higher than yesterday," draw debugging approach

**What to Show**:
- Monitoring dashboards to check
- System components to investigate
- Metrics to compare
- Hypothesis tree
- Resolution steps

---

## Common Whiteboarding Mistakes

### Mistake 1: Drawing Before Thinking

**Problem**: Start drawing immediately without planning

**Fix**:
1. Think for 30 seconds
2. Say "Let me start with high-level components"
3. Sketch mental outline
4. Then start drawing

### Mistake 2: Too Much Detail Too Soon

**Problem**: Drawing all details of first component before showing overall system

**Fix**:
- Start with boxes for all major components
- Connect them
- Then add details layer by layer

### Mistake 3: Poor Space Management

**Problem**: Run out of room, cramped diagram

**Fix**:
- Plan layout mentally first
- Start in upper-left (leave room to expand)
- Keep components evenly sized
- Use full board/canvas

### Mistake 4: Not Narrating

**Problem**: Draw in silence, interviewer lost

**Fix**:
- Talk through every box and arrow
- Explain why you're adding each component
- Pause to check understanding

### Mistake 5: Messy, Illegible

**Problem**: Sloppy handwriting, overlapping arrows

**Fix**:
- Write clearly, larger than you think
- Use straight lines (virtual: snap to grid)
- Leave space between components
- Erase/redo if messy (better than leaving it)

---

## Whiteboarding Checklist

**Before You Start**:
- [ ] Understand the problem fully (ask clarifying questions)
- [ ] Plan your layout mentally
- [ ] Identify main components
- [ ] Know what level of detail is expected

**While Drawing**:
- [ ] Narrate as you draw
- [ ] Check for understanding frequently
- [ ] Use clear, consistent notation
- [ ] Leave space for additions
- [ ] Engage with interviewer's questions

**After Main Diagram**:
- [ ] Step back and review
- [ ] Explain flow end-to-end
- [ ] Highlight key design decisions
- [ ] Invite questions
- [ ] Be ready to drill into details

---

## Physical Whiteboard Tips

**Supplies**:
- Use multiple colors (but not more than 3-4)
- Good eraser (clean, not smudgy)
- Fine-tip markers (readable from distance)

**Technique**:
- Stand to the side (don't block view)
- Write larger than you think necessary
- Use full board (don't huddle in corner)
- Step back periodically to review

**Practice**:
- Buy small whiteboard for home practice
- Practice at actual whiteboard before interview
- Get comfortable with physical writing

---

## Real Interview Examples

### Example 1: On-Site Whiteboard

**Scenario**: "Design a system to serve 10K RPS for a 70B model"

**Interviewer Observation**:
"Candidate started by drawing user, then load balancer, then API layer. Explained each component before moving to next. Used clear boxes and arrows. Added monitoring and caching systems after core flow. Annotated key numbers (batch size, GPU count). When I asked about fault tolerance, they easily added redundancy to existing diagram. Clear, structured, interactive."

**Result**: Strong hire

### Example 2: Virtual Whiteboard

**Scenario**: "Explain how PagedAttention reduces memory fragmentation"

**Interviewer Observation**:
"Candidate used Excalidraw. Started with simple example showing traditional approach's waste. Then introduced blocks and showed block table. Used colors to highlight which blocks belong to which sequence. Walked through concrete example of accessing token 35. Clear visual comparison of memory utilization. Engaged me by asking if I wanted to see the CUDA implementation next."

**Result**: Strong hire

---

## Practice Routine

**Week 1-2: Fundamentals**
- Practice basic shapes and flows
- Draw 5 system diagrams
- Record and review

**Week 3: Real Problems**
- Practice 10 interview-style questions
- Focus on clarity and speed
- Get feedback from peer

**Week 4: Mock Interviews**
- Full whiteboard sessions
- Time pressure
- Incorporate feedback

---

**Remember**: Whiteboarding is a skill that improves with practice. The more you do it, the more natural it becomes. Your goal is to make complex ideas clear, not to create perfect art.

**You've got this! 🎨**

---

*Last Updated: 2025-11-19*
*Module 9: Whiteboarding Practice Guide*
