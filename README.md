# Political Ideology of Gen Z in Germany
Inspired by the articale [Is the ideology gap growing?](https://www.allendowney.com/blog/2024/01/28/is-the-ideology-gap-growing/) by [Allen Downey](https://www.allendowney.com/wp/) I decided to do a similar (simple) analysis for Germany. The variable used is `pa01` which is the self reported political ideology. The data is from the German General Social Survey (ALLBUS) and the analysis is based on the cumulative data up to 2018 and the cross sectional data for 2021.`

## Data
The data is based on the German *ALLBUS* survey ( German General Social Survey). The data is available at the [GESIS Data Archive](https://www.gesis.org/en/allbus/allbus-home). The data is available for free, but you need to register. 

You will need two datasets:
- The cumulative data up to 2018 which you can find here: [ALLBUS 1980-2018 Cumulative File](https://search.gesis.org/research_data/ZA5276)
- The cross sectional data file for 2021 which you can find here: [ALLBUS 2021](https://search.gesis.org/research_data/ZA5282)

Download the SPSS (`*.sav``) datasets and put them into a `data/` folder within the project.

## Usage
This project uses poetry for dependency management. To install the dependencies run:
```bash
poetry install
```

To execute the analysis run:
```bash
poetry run python plot.py
```

The results will be saved in the `produc# Political Ideology of Gen Z in Germany
Inspired by the articale [Is the ideology gap growing?](https://www.allendowney.com/blog/2024/01/28/is-the-ideology-gap-growing/) by [Allen Downey](https://www.allendowney.com/wp/) I decided to do a similar (simple) analysis for Germany.

## Data
The data is based on the German *ALLBUS* survey ( German General Social Survey). The data is available at the [GESIS Data Archive](https://www.gesis.org/en/allbus/allbus-home). The data is available for free, but you need to register. 

You will need two datasets:
- The cumulative data up to 2018 which you can find here: [ALLBUS 1980-2018 Cumulative File](https://search.gesis.org/research_data/ZA5276)
- The cross sectional data file for 2021 which you can find here: [ALLBUS 2021](https://search.gesis.org/research_data/ZA5282)

Download the SPSS (`*.sav``) datasets and put them into a `data/` folder within the project.

## Usage
This project uses poetry for dependency management. To install the dependencies run:
```bash
poetry install
```

To execute the analysis run:
```bash
poetry run python plot.py
```

The results will be saved in the `produc