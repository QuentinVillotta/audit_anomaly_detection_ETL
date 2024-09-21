# audit_anomaly_detection

## Overview

As a humanitarian research organisation, IMPACT Initiatives is conducting a large amount of structured data collection in over 30 crisis affected countries, a lot of which involves large-scale nationwide data collection, including inaccessible and hard-to-reach contexts. 

One of the main issues of collecting data in conflict zones, including inaccessible and hard-to-reach areas, is that the direct control over the work of those who physically collect the data (enumerators) is very complex. The reality of things is that, for several logistical and other reasons, it can happen rather commonly and unfortunately that data is faked or incorrectly entered by enumerators. It is therefore possible that entire interviews do not necessarily reflect the views of an interviewed household member but are answered randomly by the enumerators. 

The difficulties in identifying such falsified data can arise from different aspects, and while at times they can be spotted successfully, and caught at early stages of data collection, in other cases it is more difficult, and can result in a posteriori cleaning. This may be overcorrecting for wrongly collected data (for example, deleting all entries for a specific enumerator if suspicious, losing large chunks of information, and taking a – correct – conservative approach). 
For all of its quantitative data collection, IMPACT initiatives used a data collection tool called KoboToolbox based on a tool called ODK (open data kit). In the past years, ODK and KoboToolbox introduced a functionality called audit that record how the survey form is filled. The audit files are always structured in the same format for all the surveys conducted with KoboToolbox, even though the questionnaires are different. For some time IMPACT have used the data coming from the audit tool on a very ad hoc basis to assist the data cleaning but the need for more holistic approach was identified. 

In collaboration with the Hack4Good student initiative at the Swiss Federal Institute of Technology in Zurich (ETH), IMPACT Initiatives has been exploring machine learning solutions . Specifically, the collaboration looked at the use of   Kobo   audit   functionality to   collect   data   on   enumerator   behaviour,   and   by analysing the data from the 2021 and 2022 MSNAs, students successfully identified anomalies   and   flagged   suspicious   surveys.   For   instance,   if   an   enumerator consistently submitted surveys with long pauses during interviews, it could indicate erroneous activity. The students employed the Isolation Forest algorithm to detect these anomalies. Following up from this, IMPACT’s global Research team, with the support of Unit 8,   aims to expand this pilot project to encompass a broader range of data collection exercises, and expand the scope of deployment within IMPACT. 

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```
## 
We have two data connection set up one to work with local files and one to work with kobo files. The result of both are identical. It is recommended to work with the connection to kobo api to always have the algorithm process the newest data.   

### Running Project with data in folders

1. Create a subfolder in /data/01_raw/ with the {project_short} name you want. Ex: MSNA_2023_SYR

2.  Make the following changes to your data

- rename the dataset to raw_data.xlsx
- place raw_data.xlsx in /data/01_raw/{project_short}/ 
- rename the questionnaire file to raw_questionnaire.xlsx 
- place raw_questionnaire in /data/01_raw/{project_short}/ 
  
3. Unzip the audit directory and place in /data/01_raw/{project_short}/files/

4. create file conf/files/globals.yml, copy the content below & fill out the according fields, the columns refer to the names of the columns in your file:

```
project_short: ...                 #(example: MSNA_2023_KEN)

sheet_name: ...                    #(example: KEN2302_MSNA, refers to first sheet in the MSNA file, in the case of KEN: )

data_columns:
  start: ...              #(example: start, start_time... it refers to the column where the start date is listed)
  enum_id: ...             #(example: enum_id, i_enum_id, enum_code, refers to column where the id of the enumerator is listed)
  audit_id: ...            #(example: audit_id, audit_URL, refers to column where the unique identifier of survey is listed)

audit_columns: (usually these columns remain consistent, else change to fit your file)
  event: event       
  node: node
  start: start
  end: end
  old-value: old-value
  new-value: new-value

questionnaire_columns: #(usually these columns remain consistent, else change to fit your file)
  name: name
  constraint: constraint
  type: type
  relevant: relevant
```

5.  Execute script
```
kedro run --env files
```


### Running Project with KoboToolbox API

You can run your Kedro project with:

1. create file conf/api/credentials.yml

    ```
    kobo_credentials: "Token {insert token}"
    ```

2. create file conf/api/globals.yml

kpi depends on your account being on european/international servers.
For european it's eu.kobotoolbox.org

```
asset_uid: ...
kobo_server: ...

url: https://${kobo_server}/api/v2/assets/${asset_uid}/data/?format=json
url_questionnaire: https://${kobo_server}/api/v2/assets/${asset_uid}/?format=json

project_short: ...                 (example: MSNA_2023_KEN)

data_columns:
  start: ...              #(example: start, start_time... it refers to the column where the start date is listed)
  enum_id: ...            # (example: enum_id, i_enum_id, enum_code, refers to column where the id of the enumerator is listed)
  audit_id: ...            #(example: audit_id, audit_URL, refers to column where the unique identifier of survey is listed)

audit_columns: #(usually these columns remain consistent, else change to fit your file)
  event: event       
  node: node
  start: start
  end: end
  old-value: old-value
  new-value: new-value

questionnaire_columns: #(usually these columns remain consistent, else change to fit your file)
  name: name
  constraint: constraint
  type: type
  relevant: relevant
```



3. 
```
kedro run --env api
```

If you want to run the full pipeline fetching new data you should run 
    ```
    kedro run 
    ```
    However, it can be useful to store the files locally when you want to experiment with different processing pipelines. In that case you should run 
    ```
    kedro run --pipeline data_download
    ``` 
    
    Afterwards, you can experiment with different data processing and model training pipelines by running. 
    ```
    kedro run --pipeline data_processing
    ```
    and 
    ```kedro run --pipeline data_science```
## How to test your Kedro project

Have a look at the files `src/tests/test_run.py` and `src/tests/pipelines/test_data_science.py` for instructions on how to write your tests. Run the tests as follows:

```
pytest
```

To configure the coverage threshold, look at the `.coveragerc` file.

## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. Install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
