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


