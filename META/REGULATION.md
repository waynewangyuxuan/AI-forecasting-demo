# Development Regulations
This document outlines the core principles that I, the AI assistant, must follow during the development of the `Life2Notion` project. These rules ensure the codebase is maintainable, scalable, and easy to navigate. 

## 1. Mobile Dev Best Practice
A general guideline is, we are working on a apple (MacOs, iOS, etc development), so we will have the best practice in Apple related mobile development.

## 2. Principle of Atomicity
This is the foundational principle of our project. It applies to both file structure and code.

### 2.1. Atomic File Structure
- Each file should have a single, well-defined purpose.
- A file that defines a server should only define the server. A file that contains utility functions should only contain those functions.
- Avoid creating monolithic files that handle multiple, unrelated responsibilities.

### 2.2. Atomic Code (Functions/Classes)
- Every function or class should do one thing and do it well.
- Functions should be small and focused. If a function is performing multiple distinct operations, it should be broken down into smaller, more specific functions.
- This principle promotes reusability and simplifies testing and debugging.

## 3. Always Test
- Always implement comprehensive, reasonable, and well-defined test cases before we move too far on any component/feature development.
- Flexiblely using those tests during development to verify coding idea/logic.
- Tests should not be overwelming; tests should not intefere the developmemnt; tests should be helping the development.

## 4. Principle of Co-located Documentation
To ensure that the project remains understandable and that the purpose of complex components is clear, we will adhere to the following documentation rule.
- Any significant feature, module, or service within a package must be accompanied by a `***_META.md` file in the same directory, where `***` is our component/feauture name.
- This file should explain the feature's purpose, its core logic, and how it interacts with other parts of the system.
- **Example**: If we create a complex `PropertyRecommender` service within the `agent` package, it must be located alongside a `PropertyRecommender_METAr.md` file that explains its algorithm and usage.

## 5. Proper File Structure
- Files Serve different functionality at different level of our service should be put into different folder or sub-folder, with a name of the folder that can explain its content concisely.
- Working with the `Principle of Co-located Documentation`, each folder should have one and only one META.md that describes all of its content, like scripts, sub-folders
- Properly use sub-folders to organized our code, understand our file structure as a tree, and any non-leaf node should not have too many files, or no file at all if logically there is no file at the abstract level.

## 6. Comments and Code Style
- Adapt Google Style in specific code writing
- Comment should be good so that another developer would be able to pickup the development the project.

## 7. Iteration Awareness
- We code in iterations. Therefore, it is extremely important to be aware of clean up necessary old-version code if the older versions are no longer in use.
- Avoid redundency at your best.