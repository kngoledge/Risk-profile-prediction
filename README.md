The purpose of this project is to develop a prediction system for issues of a potential investment for a multilateral-finance project; this machine learning algorithm will read input of sector and region of interested investment and make risk profile predictions on the directional change of issues given the profiles of similar investments and existing project complaint data. The input is a two variable tuple of string for sector and string for region; this input expresses investment interest for a specific sector in a specific region. The output is a dictionary of strings mapped onto floats with the key as the name of an issue that could arise from the investment of such a project, and the associated value as the probability of that issue’s occurrence. 

This project uses datasets on public financing operations and capital projects financed by international financial institutions such as the World Bank, Inter-American Development Bank, and European Investment Bank. These datasets are free and publicly available as mandated by public disclosure policy, and they are found at the websites of the institutions which finance them. These datasets contain standardized information on project name, description, region, country, sector, and financing operations. Layered within these datasets for our project is a dataset offered by the international nonprofit Accountability Counsel, which contains information about filed, registered, and closed complaints against internationally financed-projects; specifically, it contains standardized data on names of projects which have had or are currently processing complaints, descriptions of projects and complaints, regions of projects, their countries, and their issues. Another layer of information the project uses is the socio-economic profiling achieved through sets of geospatial data on population density, agricultural farmland area, water reservoir area, and per capita GDP, sourced from the online sources and databases of the International Justice Resource Center and by World Bank Open Data, each of which are applied to form a comprehensive list of similar regions to achieve a holistic measurement of potential issues. 

The complaint data is split into testing and training datasets. Success is measured based on how many outputs from the testing dataset are correctly classified. To measure this, we will compute the frequencies of occurrences of issues for each sector and region in our dataset and use those frequencies to estimate probability, then compare those probabilities against our predictions.

Our preliminary data is from the international nonprofit Accountability Counsel, which provides information on a project (i.e. location, sector) and its associated complaints. Based on this information, our project will be able to caution investors about potential issues that could arise with their proposed project. An example input would be “(Energy sector, Cameroon)” with a goal to unearth the possible risks associated with a new project in the Energy sector for the country of Cameroon. The output would state potential issues and their probabilities, and would look as such: “{(Consultation and disclosure: 0.25), (Due diligence: 0.14), (Labor: 0.05), (Environmental: 0.01)}”.

The baseline is to return the single most common issue that arise in the set of issues from the exact matches of projects with the same sector and region. Given an input, the baseline algorithm looks at projects with the exact matches to the input, and outputs the issue that has the highest frequency. The output for the baseline algorithm for input ('Forestry', 'Uganda') was ‘Displacement (physical and/or economic)’, meaning that forestry projects that have received complaints in Uganda have had ‘Displacement (physical and/or economic)’ as its most frequent issue.

The oracle employs our team members to manually classify example inputs and determine the three most likely potential issues per input. The agreement rate among our results for 30 examples was found to be 75.5% (2). Whereas the baseline only outputs one issue per input, the oracle provides greater insight in issues that may still be likely to occur at relatively lower probabilities. Overall, the oracle would be more extensive in terms of output, because the outputs for the baseline would be none or one element. 

Challenges exist within the datasets themselves. The size of the complaint data contains about 816 data points, which is small; however, this data set will be split 616/200 to be used for training/testing (1). The training set will also include project data information which will then be upwards to 20,000 data points. Labelling multiple issues is also a challenge. Each training point belongs to a different classes of issues; ideally we construct a function which, given a new data point, will correctly predict the issues to which the new point belongs. Relevant topics for the project would be multilabel classification, or perhaps multiclass classification or an adapted k-nearest neighbors algorithm for multilabel classification. Other relevant topics would be semi-supervised learning algorithms for a logistic classifier, especially given the simultaneously labeled and unlabeled data with regards to issues classification.

In terms of related work, Real Capital Analytics is a data and analytics company focusing on the investment market for commercial real estate(3). Their products utilize a large database of commercial real estate transactional information and then analyze these large datasets to provide insight on commercial real estate investment and discover trends for commercial property, which is similar to our project’s intended behavior. Likewise, RepRisk is a company that helps to identify and assess environmental, social, and corporate governance (ESG) and business conduct risks. RepRisk facilitates risk management by providing an initial risk screening and daily monitoring of risks, which is similar to our project’s intention of informing investors of potential risks associated with various financed project ideas. 
