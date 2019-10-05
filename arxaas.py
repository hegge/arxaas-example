#!/usr/bin/env python3

# Prerequisites:
# Python 3.6+
# Docker

# Set-up:
# docker pull navikt/arxaas
# docker run -p 127.0.0.1:8080:8080/tcp navikt/arxaas
# pip install arxaas

from pyarxaas import ARXaaS
from pyarxaas import Dataset
from pyarxaas import AttributeType
from pyarxaas.privacy_models import KAnonymity, LDiversityDistinct
from pyarxaas.hierarchy import IntervalHierarchyBuilder
import pandas as pd

# generate sample data file
with open('data.csv', 'w+') as f:
    f.write('age;gender;id;name;disease\n')
    f.write('34;male;81667;Per;flu\n')
    f.write('45;female;81675;Ada;bronchitis\n')
    f.write('66;male;81925;Pal;pneumonia\n')
    f.write('70;female;81933;Gia;gastritis\n')
    f.write('34;female;81931;Joy;gastric ulcer\n')
    f.write('70;male;81934;Espen;stomach cancer\n')
    f.write('45;male;81932;Per;flu\n')

# establish a connection to the ARXaaS service
arxaas = ARXaaS("http://localhost:8080")

# read data
df = pd.read_csv("data.csv", sep=";")
dataset = Dataset.from_pandas(df)
print("dataset:")
print(df)

# set attribute type
dataset.set_attribute_type(AttributeType.QUASIIDENTIFYING, 'gender', 'age')
dataset.set_attribute_type(AttributeType.IDENTIFYING, 'id', 'name')
dataset.set_attribute_type(AttributeType.SENSITIVE, 'disease')

# get the risk profle of the dataset
risk_profile = arxaas.risk_profile(dataset)

# get risk metrics
re_indentifiation_risk = risk_profile.re_identification_risk
distribution_of_risk = risk_profile.distribution_of_risk
print("re_indentifiation_risk: " + str(re_indentifiation_risk))
#print("distribution_of_risk: " + str(distribution_of_risk))

# generate sample hierarchy file
with open('gender_hierarchy.csv', 'w+') as f:
    f.write('male;*\n')
    f.write('female;*\n')

# import the hierarchy from a local csv file
gender_hierarchy = pd.read_csv("gender_hierarchy.csv", header=None, sep=";")

# create interval based hierarchy
hierarchy_builder = IntervalHierarchyBuilder()
hierarchy_builder.add_interval( 0,  16, "child")
hierarchy_builder.add_interval(16,  40, "young-adult")
hierarchy_builder.add_interval(40,  60, "adult")
hierarchy_builder.add_interval(60, 120, "old")
ages = df['age'].to_list() # extract ages from dataset
age_hierarchy = arxaas.hierarchy(hierarchy_builder, ages)

# assign hierarchy to attributes
dataset.set_hierarchy('gender', gender_hierarchy)
dataset.set_hierarchy('age', age_hierarchy)

# create privacy models
kanon = KAnonymity(1)
ldiv = LDiversityDistinct(2, "disease")

# anonymize the dataset using a list of privacy models
anonymize_result = arxaas.anonymize(dataset, [kanon, ldiv], 0.2)

# get the anonymous dataset
anonymized_dataset = anonymize_result.dataset
print("\nanonymized_dataset:")
print(anonymized_dataset.to_dataframe())
print(anonymize_result.anonymization_status)

# get the risk profile for the anonymous dataset
anon_risk_profile = anonymize_result.risk_profile

# get risk metrics as a dictionary
anon_re_indentifiation_risk = anon_risk_profile.re_identification_risk
anon_distribution_of_risk = anon_risk_profile.distribution_of_risk
print("anon_re_indentifiation_risk: " + str(anon_re_indentifiation_risk))
#print("anon_distribution_of_risk: " + str(anon_distribution_of_risk))

# get the anonymiztion metrics
anon_metrics = anonymize_result.anonymization_metrics
print("\nanon_metrics:")
print("attribute_generalization: " + str(anon_metrics.attribute_generalization))
