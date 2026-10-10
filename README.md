# kaggleComp

Shared repo for the Kaggle Titanic competition (CS 4320).

## Data

- `train.csv` - 891 labeled passengers. Target column is `Survived` (0/1).
  Features: `PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked`.
- `test.csv` - 418 unlabeled passengers used to generate the submission.

## Submission format

A CSV with exactly two columns, one row per `PassengerId` in `test.csv`:

```
PassengerId,Survived
892,0
893,1
```

Upload at https://www.kaggle.com/competitions/titanic/submit

## Conventions

- Keep generated files (`submission*.csv`, model artifacts, notebook checkpoints) out of git; the `.gitignore` already covers them.
- Put your own experiments in a notebook or script named after yourself or the approach so work doesn't collide.
