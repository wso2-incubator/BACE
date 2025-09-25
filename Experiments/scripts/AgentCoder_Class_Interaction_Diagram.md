# AgentCoder Class Interaction Diagram

## 🏗️ **Architecture Overview**

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AgentCoder System Architecture                        │
└─────────────────────────────────────────────────────────────────────────────────┘

                                   ┌──────────────────┐
                                   │ AgentCoderConfig │◄──── Configuration
                                   │ (dataclass)      │       Input
                                   └─────────┬────────┘
                                            │
                                            ▼
                                  ┌──────────────────┐
                                  │ AgentCoderGen-   │◄──── Main Orchestrator
                                  │ erator           │       (Entry Point)
                                  └─┬──┬──┬──┬──┬───┘
                                    │  │  │  │  │
          ┌─────────────────────────┘  │  │  │  └─────────────────────┐
          │                            │  │  │                        │
          ▼                            │  │  │                        ▼
    ┌──────────┐                      │  │  │                 ┌──────────────┐
    │FileManager│                      │  │  │                 │CompletionGen-│
    │          │                      │  │  │                 │erator        │
    └─────┬────┘                      │  │  │                 └──────┬───────┘
          │                           │  │  │                        │
          │ manages files             │  │  │                        │ coordinates
          │                           │  │  │                        │ generation
          ▼                           │  │  │                        ▼
    ┌──────────────┐                 │  │  │                 ┌──────────────┐
    │FinalCompletion│                 │  │  │                 │IterationResult│
    │(dataclass)   │                 │  │  │                 │(dataclass)    │
    └──────────────┘                 │  │  │                 └───────┬──────┘
                                     │  │  │                         │
                                     ▼  │  │                         ▼
                              ┌─────────────┐                ┌──────────────┐
                              │IterationTracker│              │TestResults-   │
                              │             │                │Summary       │
                              └─────┬───────┘                │(dataclass)   │
                                    │                        └──────────────┘
                                    │ tracks iterations
                                    │
                                    ▼
                              ┌─────────────┐
                              │WorkflowBuilder│
                              └─────┬───────┘
                                    │ uses prompts
                                    ▼
                              ┌─────────────┐
                              │PromptProvider│
                              └─────────────┘
```

## 🔄 **Detailed Class Relationships**

### **1. Core Orchestration Layer**

```text
AgentCoderGenerator (Main Controller)
├── Initializes: FileManager, IterationTracker, WorkflowBuilder, CompletionGenerator  
├── Coordinates: Problem processing workflow
├── Delegates: File I/O, iteration tracking, workflow execution
└── Returns: Final completion results
```

### **2. Configuration & Data Layer**

```text
AgentCoderConfig ──┬─► FileManager
                   ├─► IterationTracker  
                   ├─► WorkflowBuilder
                   └─► AgentCoderGenerator

TestResultsSummary ──► IterationResult ──► IterationTracker

FinalCompletion ──► FileManager
```

### **3. Workflow Processing Layer**

```text
WorkflowBuilder
├── Uses: PromptProvider (for all prompt generation)
├── Creates: LangGraph StateGraph workflow
├── Manages: Agent nodes (programmer, test_designer, test_executor)
└── Returns: Compiled workflow

CompletionGenerator  
├── Executes: Workflow with stream mode
├── Processes: Workflow updates (programmer, test_executor)
├── Creates: IterationResult objects
└── Delegates: Saving to IterationTracker
```

### **4. File Management Layer**

```text
FileManager
├── Manages: Output file paths and directory structure
├── Handles: Final completion saving
└── Provides: Path generation utilities

IterationTracker
├── Uses: FileManager (for path management)
├── Creates: IterationResult objects  
├── Saves: Per-iteration results to files
└── Manages: Early completion file duplication
```

## 🔀 **Data Flow Diagram**

```text
[HumanEval Problem]
        │
        ▼
┌──────────────────┐
│AgentCoderGen-    │
│erator.process_   │     ┌─────────────────┐
│all_problems()    │────►│FileManager      │
└─────────┬────────┘     │.clear_output_   │
          │              │files()          │
          │              └─────────────────┘
          ▼
┌──────────────────┐
│CompletionGen-    │
│erator.generate_  │
│completion_with_  │
│iterations()      │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐     ┌─────────────────┐
│WorkflowBuilder   │────►│PromptProvider   │
│.create_initial_  │     │.get_programmer_ │
│state()           │     │prompt()         │
└─────────┬────────┘     │.generate_failure│
          │              │feedback()       │
          │              └─────────────────┘
          ▼
┌──────────────────┐
│Workflow.stream() │
│(LangGraph)       │
└─────────┬────────┘
          │
    ┌─────▼─────┐
    │  Updates  │
    │Processing │
    └─────┬─────┘
          │
          ▼
┌──────────────────┐     ┌─────────────────┐
│IterationTracker │────►│TestResultsSummary│
│.create_iteration │     │(dataclass)      │
│result()          │     └─────────────────┘
└─────────┬────────┘
          │
          ▼
┌──────────────────┐     ┌─────────────────┐
│IterationTracker │────►│IterationResult  │
│.save_iteration_  │     │.to_dict()       │
│result()          │     └─────────────────┘
└─────────┬────────┘
          │
          ▼
┌──────────────────┐     ┌─────────────────┐
│FileManager       │────►│FinalCompletion  │
│.save_final_      │     │.to_dict()       │
│completion()      │     └─────────────────┘
└──────────────────┘
```

## 🎯 **Key Design Patterns**

### **1. Dependency Injection**

```text
AgentCoderGenerator
├── Injects FileManager into IterationTracker
├── Injects config into all components
└── Injects workflow into CompletionGenerator
```

### **2. Single Responsibility**

```
```

### **2. Single Responsibility**

```text
PromptProvider    → Only handles prompt generation
FileManager       → Only handles file operations  
IterationTracker  → Only handles iteration management
WorkflowBuilder   → Only handles workflow construction
CompletionGenerator → Only handles completion processing
```

### **3. Data Transfer Objects**

```text
AgentCoderConfig   → Configuration container
TestResultsSummary → Test result container
IterationResult    → Iteration data container
FinalCompletion    → Final result container
```

## 🔗 **External Dependencies**

```text
┌─────────────────┐     ┌─────────────────┐
│ LangChain/      │────►│ WorkflowBuilder │
│ LangGraph       │     │                 │
└─────────────────┘     └─────────────────┘

┌─────────────────┐     ┌─────────────────┐
│ HumanEval       │────►│ AgentCoder-     │
│ Dataset         │     │ Generator       │
└─────────────────┘     └─────────────────┘

┌─────────────────┐     ┌─────────────────┐
│ Common Module   │────►│ Multiple        │
│ (CodeProcessor, │     │ Classes         │
│  Sandbox)       │     └─────────────────┘
```

## 📊 **Class Complexity Metrics**

| Class | Lines | Responsibilities | Dependencies |
|-------|-------|------------------|--------------|
| `AgentCoderGenerator` | ~50 | Orchestration only | All others |
| `CompletionGenerator` | ~70 | Workflow execution | Workflow, IterationTracker |
| `WorkflowBuilder` | ~120 | LangGraph workflow | PromptProvider |
| `PromptProvider` | ~60 | Prompt generation | None |
| `IterationTracker` | ~40 | Iteration management | FileManager |
| `FileManager` | ~30 | File operations | None |

This refactored architecture achieves excellent **separation of concerns**, **testability**, and **maintainability**! 🚀
