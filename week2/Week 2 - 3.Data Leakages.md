# Data Leakage

- Leakage types and examples
- Competition specific. Leaderboard probing
- Concrete walkthroughs

## Basic data leaks

> Data Leaks are unexpected errors that expose extra information that wouldn’t be available in production.

### Leaks in time series

- Split should be done on time
  - In real life we don’t have information from future
  - In competitions first thing to look: train/public/private
- Even when split by time, features may contain information about future
  - User history in CTR tasks
  - Weather

### Unexpected information

- Meta data (file creation date, image resolution)
- Information in IDs 
- Row order



## Leaderboard probing

A technique looking for dataleaks based on the leader board.



## Additional material and links



- [Perfect score script by Oleg Trott](https://www.kaggle.com/olegtrott/the-perfect-score-script) -- used to probe leaderboard
- [Page about data leakages on Kaggle](https://www.kaggle.com/docs/competitions#leakage)
- [Another page about data leakages on Kaggle](https://www.kaggle.com/dansbecker/data-leakage)