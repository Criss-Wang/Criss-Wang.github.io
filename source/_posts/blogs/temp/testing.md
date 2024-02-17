
A typical software testing suite will include:
- unit tests which operate on atomic pieces of the codebase and can be run quickly during development,
- regression tests replicate bugs that we've previously encountered and fixed,
- integration tests which are typically longer-running tests that observe higher-level behaviors that leverage multiple components in the codebase,

For machine learning systems, we should be running model evaluation and model tests in parallel.
- Model evaluation covers metrics and plots which summarize performance on a validation or test dataset.
- Model testing involves explicit checks for behaviors that we expect our model to follow.

How do you write model tests?
1. Pre-train test
	- Early bug discovery + training short-circuiting (saves training cost)
	- Things to check:
		- output distribution
		- gradient-related information (training loss curve)
		- data quality
		- label leakage
2. Post-train test
	- post mortem issue discovery and model behavior analysis
		- Things to check:
			- Invariance Test (use a set of perturbations we should be able to make to the input without affecting the model's output)
			- Directional Expectation Test
			- Data Unit Test (similar to regression test, with failued model scenarios)

3. Organizing tests
	- structuring your tests around the "skills" we expect the model to acquire while learning to perform a given task.

4. Model Dev Pipeline
	1. ![](https://www.jeremyjordan.me/content/images/size/w1000/2020/08/Group-7.png)

 {source: [https://www.jeremyjordan.me/testing-ml/](https://www.jeremyjordan.me/testing-ml/)}