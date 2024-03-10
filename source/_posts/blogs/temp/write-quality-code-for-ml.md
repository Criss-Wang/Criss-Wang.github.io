

# Introduction
Too many poor code in ml research and ml production


## 1.  Proof-of-Concept Style code
**Issue**: POC code hard to extend (5 to 10 ops rewritten to complete preprocessing, feature engineering, training, deployment and monitoring)

**Rationale**: Common in **Startup**, could be fast in the short-term, but detrimental in long-term]

**Solution**: Use library or custom packages for *argparse* and other argument management. Example: [typer](https://typer.tiangolo.com/), [FastAPI](https://fastapi.tiangolo.com/)

## 2. No high-level separation of concerns
**Issue**: number of cyclic dependencies present between what seems to be low-level packages and high-level ones increases

**Rationale**: What the ML package is doing (under the name ML lib) also include administrative code, etc

**Solution**: Use *Docker* and *Microservices* architecture. Make sure to achieve good distributed system hygiene with middlewares like RabbitMQ, Kafka and Redis
## 3. No low-level separation of concerns
**Issue**: very bad code structure. no OOP or FP

**Solution**: checkout my code architecture in "/Users/criss_w/Desktop/Research_and_ML/Self_Study/sample_full_stack_ml/model"

## 4. No configuration Data Model
**Issue**: Debugging is really a nightmare

**Solution**: Pydantic

## 5. Handling legacy models
**Issue**: When trying to achieve backward compatibility, poor coding structure give much pain

**Solution**: `cron`, `plotly`, `tmux`. Understand basic deployment strategies

## 6. Code quality: type hinting, documentation, complexity, dead code
**Solution**: `autopep8, flake, mypy, pylint, unittest, pydeps, sourcery`
