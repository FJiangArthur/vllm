# Mock Interview Guide

## Introduction

This guide provides a complete framework for conducting effective mock interviews. Mock interviews are the most valuable preparation tool—they simulate real interview pressure and help you identify weak areas.

**Goal**: Complete 5+ full mock interviews before real interviews
**Duration**: 60-90 minutes per mock interview
**Format**: With peer, mentor, or recorded self-practice

---

## Why Mock Interviews Matter

### Statistics
- **Candidates who do 5+ mocks**: 3-4x higher offer rate
- **Improvement curve**: Biggest gains between mocks 2-4
- **Confidence**: 80% report lower anxiety after practice

### Benefits
1. **Identify gaps**: Discover what you don't know
2. **Practice communication**: Learn to explain clearly
3. **Build confidence**: Reduce interview anxiety
4. **Time management**: Learn to pace yourself
5. **Handle pressure**: Get comfortable with being uncomfortable
6. **Get feedback**: Iterate and improve

---

## Mock Interview Formats

### Format 1: Full Technical Interview (60 min)

**Structure**:
```
0:00-0:05  Introduction & warm-up
0:05-0:25  Technical deep dive (concepts + coding)
0:25-0:45  System design
0:45-0:55  Behavioral question
0:55-1:00  Questions for interviewer
```

**Evaluation**: Use rubric (provided below)

---

### Format 2: Technical Deep Dive Only (45 min)

**Structure**:
```
0:00-0:05  Introduction
0:05-0:15  Concept explanation (e.g., PagedAttention)
0:15-0:30  Live coding (CUDA kernel)
0:30-0:40  Debugging scenario
0:40-0:45  Wrap-up
```

**Use case**: Focus on technical depth

---

### Format 3: System Design Only (45 min)

**Structure**:
```
0:00-0:05  Problem statement + clarifications
0:05-0:15  High-level architecture
0:15-0:30  Detailed component design
0:30-0:40  Scaling & trade-offs
0:40-0:45  Follow-up questions
```

**Use case**: Practice architectural thinking

---

### Format 4: Behavioral Only (30 min)

**Structure**:
```
0:00-0:05  Warm-up question
0:05-0:20  3-4 behavioral questions (STAR method)
0:20-0:28  Situational question
0:28-0:30  Wrap-up
```

**Use case**: Perfect your stories

---

## Setting Up Mock Interviews

### Finding Mock Interview Partners

**Options**:
1. **Peer learners**: Study group, online communities
2. **Current employees**: At target companies (via LinkedIn)
3. **Professional services**: interviewing.io, Pramp
4. **Mentors**: Former colleagues, professors
5. **Self-practice**: Record yourself on video

### Logistics

**Scheduling**:
- Book 1-2 mocks per week
- Space them 3-4 days apart (time to improve)
- Morning preferred (peak alertness)

**Tools**:
- **Video**: Zoom, Google Meet
- **Whiteboard**: Miro, Excalidraw, CoderPad
- **Code**: VSCode Live Share, CoderPad

**Ground Rules**:
- Take it seriously (no notes, no googling)
- Stick to time limits
- Honest feedback after
- Rotate roles (interviewer/candidate)

---

## Interview Rubric

### Technical Knowledge (40 points)

**Core Concepts (15 points)**:
```
[15/15] Excellent: Deep understanding, can explain to any audience
[12/15] Good: Solid knowledge, minor gaps
[9/15]  Adequate: Basic understanding, some confusion
[6/15]  Needs work: Significant gaps
[0-3/15] Poor: Fundamentally confused or incorrect
```

**CUDA/Performance (10 points)**:
```
[10/10] Can write optimized kernels, explain trade-offs
[8/10]  Good kernel knowledge, some optimization gaps
[6/10]  Basic CUDA, struggles with optimization
[4/10]  Limited CUDA knowledge
[0-2/10] Cannot write functional kernels
```

**System Design (15 points)**:
```
[15/15] Complete, scalable architecture with justifications
[12/15] Solid design, minor gaps in scaling/HA
[9/15]  Basic architecture, missing components
[6/15]  Incomplete design, flawed approach
[0-3/15] Cannot design coherent system
```

### Problem Solving (20 points)

**Approach (10 points)**:
```
[10/10] Systematic, clarifies requirements, considers alternatives
[8/10]  Good structure, mostly complete analysis
[6/10]  Some structure, jumps to solution
[4/10]  Disorganized approach
[0-2/10] Random, no clear method
```

**Coding/Implementation (10 points)**:
```
[10/10] Clean, correct, efficient code
[8/10]  Functional with minor bugs
[6/10]  Works but inefficient or buggy
[4/10]  Significant errors
[0-2/10] Cannot produce working code
```

### Communication (25 points)

**Clarity (10 points)**:
```
[10/10] Crystal clear, adapts to audience, great examples
[8/10]  Generally clear, good explanations
[6/10]  Understandable but sometimes unclear
[4/10]  Often confusing, poor structure
[0-2/10] Cannot explain coherently
```

**Collaboration (8 points)**:
```
[8/8] Asks great questions, incorporates feedback, discusses trade-offs
[6/8] Good interaction, responds to hints
[4/8] Limited interaction, misses some hints
[2/8] Poor collaboration, ignores feedback
[0-1/8] Difficult to work with
```

**Presentation (7 points)**:
```
[7/7] Engaging, confident, well-paced
[5/7] Good delivery, minor nervousness
[3/7] Adequate, some pacing issues
[1/7] Poor delivery, very nervous
[0/7] Extremely poor presentation
```

### Cultural Fit (15 points)

**Enthusiasm (5 points)**:
```
[5/5] Genuine passion for the field, excited about company
[4/5] Good interest, engaged
[3/5] Neutral, professional
[2/5] Lukewarm interest
[0-1/5] Disinterested or negative
```

**Growth Mindset (5 points)**:
```
[5/5] Handles feedback well, admits gaps, eager to learn
[4/5] Receptive to feedback
[3/5] Accepts feedback when given
[2/5] Defensive about feedback
[0-1/5] Rejects feedback, rigid
```

**Values Alignment (5 points)**:
```
[5/5] Strong alignment with company values
[4/5] Good fit
[3/5] Neutral alignment
[2/5] Some misalignment
[0-1/5] Poor fit
```

### **Total Score: /100**

**Interpretation**:
- **90-100**: Excellent, likely strong hire
- **75-89**: Good candidate, probable hire
- **60-74**: Adequate, maybe hire (with concerns)
- **40-59**: Needs improvement, likely no hire
- **<40**: Not ready for this role

---

## Question Bank for Mock Interviews

### Technical Concepts (Choose 2-3)
1. Explain PagedAttention and its benefits
2. How does continuous batching work?
3. Walk through the request lifecycle in vLLM
4. Compare vLLM to other serving frameworks
5. Explain KV cache and its memory requirements
6. Describe FlashAttention algorithm
7. How does tensor parallelism work?
8. Compare GPTQ vs AWQ quantization
9. Explain preemption in vLLM
10. How does prefix caching work?

### CUDA Coding (Choose 1)
1. Write a vector addition kernel
2. Implement parallel reduction (sum)
3. Optimize matrix transpose
4. Write softmax kernel
5. Implement layer normalization
6. Create a simple GEMM kernel
7. Write warp-level reduction
8. Implement ReLU activation fused with addition

### Debugging (Choose 1)
1. Fix uncoalesced memory access
2. Eliminate bank conflicts
3. Improve low occupancy kernel
4. Debug race condition
5. Optimize memory-bound kernel

### System Design (Choose 1)
1. Design high-throughput LLM serving (10K RPS)
2. Build multi-tenant model serving platform
3. Create cost-optimized batch inference system
4. Design ultra-low latency serving (<100ms)
5. Build streaming response system
6. Design A/B testing infrastructure

### Behavioral (Choose 2-3)
1. Tell me about a challenging bug you debugged
2. Describe a time you optimized performance significantly
3. How do you handle disagreement with teammates?
4. Tell me about a project you're proud of
5. Describe a time you had to learn something quickly
6. How do you prioritize when you have multiple urgent tasks?

---

## Mock Interview Scripts

### Script 1: Full Technical Interview

**Interviewer**: "Hi, I'm [Name], a [Role] at [Company]. I'll be your interviewer today for a GPU Systems Engineer position. We'll spend about an hour together - first some technical concepts, then a bit of coding, followed by a system design question, and we'll wrap up with a behavioral question. Sound good?"

**[5 min] Warm-up**
"Tell me about your background and what interests you about GPU systems and LLM inference."

**[20 min] Technical Deep Dive**
"Let's start with some core concepts. Can you explain how PagedAttention works and why it's important?"
- Follow-ups based on answer: block tables, memory savings, comparison to traditional attention

"Now, can you walk me through how continuous batching improves throughput?"
- Follow-up: What are the implementation challenges?

**[20 min] System Design**
"Let's shift to a design problem. Design a system to serve a 70B parameter model with 10,000 requests per second peak load and P95 latency under 500ms."
- Let candidate drive
- Ask clarifying questions they should be asking
- Probe on scaling, cost, HA

**[10 min] Behavioral**
"Tell me about a time you had to optimize code for performance. What was your approach and what was the outcome?"
- Look for STAR method: Situation, Task, Action, Result

**[5 min] Wrap-up**
"Great! Do you have any questions for me about the role or the team?"

---

### Script 2: CUDA Deep Dive

**Interviewer**: "Today we're going to focus on CUDA programming. I'll ask you to implement a few kernels and then we'll discuss optimization strategies."

**[15 min] Concept Check**
"First, can you explain the CUDA memory hierarchy and when you'd use each type of memory?"
- Follow-up: "What's the latency difference between global memory and shared memory?"
- Follow-up: "How do you avoid bank conflicts in shared memory?"

**[25 min] Live Coding**
"Let's implement a parallel reduction kernel to sum an array. You can use any approach you like."
- Watch for: Shared memory usage, synchronization, handling arbitrary sizes
- If stuck: "How would you use shared memory to reduce global memory accesses?"
- After implementation: "How would you optimize this further?"

**[15 min] Optimization**
"I'm going to share an Nsight Compute profile of a kernel. It shows 45% memory throughput, 35% occupancy, and 25% bank conflicts. How would you diagnose and fix this?"
- Look for: Systematic approach, correct diagnosis, multiple optimization strategies

---

## Feedback Framework

### Immediate Feedback (5-10 min after mock)

**Positive Feedback First**:
"Here are things you did really well:
1. [Specific strength with example]
2. [Specific strength with example]
3. [Specific strength with example]"

**Areas for Improvement**:
"Here are some areas to work on:
1. [Specific gap with example]
   → Suggestion: [How to improve]
2. [Specific gap with example]
   → Suggestion: [How to improve]
3. [Specific gap with example]
   → Suggestion: [How to improve]"

**Overall Assessment**:
"Overall score: X/100
Based on today, I think you're [ready/almost ready/need more preparation] for interviews at [company level]."

### Written Feedback Template

```
Mock Interview Feedback
Date: [Date]
Candidate: [Name]
Interviewer: [Name]
Format: [Full/Technical Deep Dive/System Design/Behavioral]

SCORES:
- Technical Knowledge: __/40
- Problem Solving: __/20
- Communication: __/25
- Cultural Fit: __/15
- Total: __/100

STRENGTHS:
1.
2.
3.

AREAS TO IMPROVE:
1.
   Action:

2.
   Action:

3.
   Action:

OVERALL ASSESSMENT:
[Free-form feedback on readiness, trajectory, recommendations]

RECOMMENDED NEXT STEPS:
[ ] Practice more [specific topic]
[ ] Review [specific material]
[ ] Schedule another mock on [topic]
[ ] Ready for real interviews!
```

---

## Self-Practice (Recording Method)

If you can't find a partner, practice alone with recording:

**Setup**:
1. Camera and screen recording
2. Whiteboard app or paper
3. Timer
4. Question list (pick randomly)

**Process**:
1. Start recording
2. Read question aloud
3. Answer as if interviewing (full explanation)
4. Solve problem on whiteboard/code
5. Stop recording after time limit

**Review (Next day)**:
1. Watch yourself objectively
2. Note: Clarity, pacing, filler words, body language
3. Identify gaps in technical accuracy
4. List improvements
5. Re-record same question after improvement

**Self-Evaluation Checklist**:
```
[ ] Did I structure my answer clearly?
[ ] Did I use concrete examples?
[ ] Did I speak at a good pace?
[ ] Did I show enthusiasm?
[ ] Was I technically accurate?
[ ] Did I handle uncertainty well?
[ ] Would I hire myself based on this?
```

---

## Iteration Strategy

### After Mock #1
**Focus**: Identify biggest gaps
**Action**: Deep study on weak areas
**Next mock**: 3-4 days later

### After Mock #2
**Focus**: Refine communication
**Action**: Practice explaining out loud
**Next mock**: 3-4 days later

### After Mock #3
**Focus**: Build confidence
**Action**: Practice comfortable topics
**Next mock**: 3-4 days later

### After Mock #4
**Focus**: Polish and consistency
**Action**: Full simulation (timed, no help)
**Next mock**: 2-3 days later

### After Mock #5+
**Focus**: Fine-tuning
**Action**: Company-specific preparation
**Ready**: Schedule real interviews!

---

## Common Pitfalls in Mock Interviews

### For Candidates

❌ **Taking it too casually**: "It's just practice"
✓ **Treat as real**: Full effort, no cheating

❌ **Giving up when stuck**: Silent for minutes
✓ **Think out loud**: Share your thought process

❌ **Defensive about feedback**: Arguing with interviewer
✓ **Receptive**: Thank them, take notes

❌ **Not asking questions**: Assuming requirements
✓ **Clarify**: Ask questions like a real interview

### For Interviewers

❌ **Too easy on feedback**: "You did great!"
✓ **Honest and specific**: Point out actual issues

❌ **Too harsh**: Discouraging
✓ **Constructive**: Positive framing, actionable

❌ **Not tracking time**: Running over
✓ **Time discipline**: Stick to schedule

❌ **Not taking notes**: Vague feedback
✓ **Document**: Write specific examples

---

## Progress Tracking

### Mock Interview Log

```
| # | Date | Format | Partner | Score | Key Learnings | Next Focus |
|---|------|--------|---------|-------|---------------|------------|
| 1 | | | | /100 | | |
| 2 | | | | /100 | | |
| 3 | | | | /100 | | |
| 4 | | | | /100 | | |
| 5 | | | | /100 | | |
```

### Skill Progression Tracking

```
Skill Area            | Mock 1 | Mock 2 | Mock 3 | Mock 4 | Mock 5 | Target
----------------------|--------|--------|--------|--------|--------|--------
Core Concepts         | __/15  | __/15  | __/15  | __/15  | __/15  | 13+/15
CUDA Knowledge        | __/10  | __/10  | __/10  | __/10  | __/10  | 8+/10
System Design         | __/15  | __/15  | __/15  | __/15  | __/15  | 12+/15
Problem Solving       | __/20  | __/20  | __/20  | __/20  | __/20  | 16+/20
Communication         | __/25  | __/25  | __/25  | __/25  | __/25  | 20+/25
Cultural Fit          | __/15  | __/15  | __/15  | __/15  | __/15  | 12+/15
```

---

## Conclusion

Mock interviews are your secret weapon. Treat them seriously, seek honest feedback, and iterate. By your 5th mock, you should feel confident and ready.

**Remember**:
- Quality > Quantity: 5 serious mocks > 20 casual ones
- Feedback is gold: Seek it actively, implement it
- Iterate: Each mock should show measurable improvement
- Stay positive: Everyone struggles at first

**You've got this! 💪**

---

*Last Updated: 2025-11-19*
*Module 9: Mock Interview Guide*
