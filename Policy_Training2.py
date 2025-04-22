import os
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss

# Load dataset
dataset = load_dataset('csv', data_files={
    "train": 'Policy_Details_Train.csv',
    "test": 'Policy_Details_Test.csv'
})

# Encode labels
le = LabelEncoder()
intent_dataset_train = le.fit_transform(dataset["train"]['Label'])
dataset["train"] = dataset["train"].remove_columns("Label").add_column("Label", intent_dataset_train).cast(dataset["train"].features)

intent_dataset_test = le.fit_transform(dataset["test"]['Label'])
dataset["test"] = dataset["test"].remove_columns("Label").add_column("Label", intent_dataset_test).cast(dataset["test"].features)

model_id = "sentence-transformers/all-mpnet-base-v2"
model = SetFitModel.from_pretrained(model_id)

trainer = SetFitTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    batch_size=64,
    num_iterations=20,
    num_epochs=2,
    column_mapping={"Inputs": "text", "Label": "label"}
)

# Train the model
trainer.train()

# Evaluate the model
evaluation_results = trainer.evaluate()
print("Evaluation Results:", evaluation_results)

os.makedirs('ckpt/', exist_ok=True)

trainer.model._save_pretrained(save_directory="ckpt/")

class_label_map = {
    "0": "Economy > General Economic Policy > Autumn Budget 2024",
    "1": "Economy > General Economic Policy > Sustained Economic Growth",
    "2": "Economy > Cost of Living > Energy Price Caps",
    "3": "Economy > Cost of Living > Household Expenses",
    "4": "Economy > Trade Policy > Rebuilding EU Trade Relations",
    "5": "Economy > Taxation > Small Business Tax Relief",
    "6": "Economy > Taxation > National Insurance Reform",
    "7": "Economy > Industrial Strategy > National Wealth Fund",
    "8": "Economy > Clean Energy > Energy Transition",
    "9": "Economy > Housing and Infrastructure > New Homes",
    "10": "Economy > Critical Perspectives > Economic Reforms",
    "11": "Public Services > Healthcare > NHS Workforce Expansion",
    "12": "Public Services > Healthcare > Mental Health Services",
    "13": "Public Services > Healthcare > Primary Care Access",
    "14": "Public Services > Education > Teacher Recruitment",
    "15": "Public Services > Education > School Infrastructure",
    "16": "Environment > Renewable Energy > Solar and Wind Power",
    "17": "Environment > Net Zero Policies > Climate Change Targets",
    "18": "Environment > Biodiversity and Conservation > Natural Habitat Protection",
    "19": "Environment > Water Management > Clean Rivers and Lakes",
    "20": "Environment > Energy Efficiency > Green Homes Initiative",
    "21": "Welfare > Housing > Affordable Housing",
    "22": "Welfare > Housing > Social Housing",
    "23": "Welfare > Benefits System > Universal Credit Reform",
    "24": "Welfare > Childcare and Family Support > Subsidized Childcare",
    "25": "Welfare > Childcare and Family Support > Family Benefits",
    "26": "Defense and Security > Military Spending > Modernization",
    "27": "Defense and Security > Cybersecurity > Threat Management",
    "28": "Defense and Security > National Security > Terrorism and Threats",
    "29": "Defense and Security > Border Security > Immigration and Safety",
    "30": "Technology > Digital Infrastructure > Broadband Access",
    "31": "Technology > Research and Development > AI and Innovation",
    "32": "Technology > Digital Skills > Workforce Training",
    "33": "Technology > Data Protection and Privacy > Regulation",
    "34": "Culture > Arts Funding > Education Support",
    "35": "Culture > Heritage Preservation > Landmark Protection",
    "36": "Culture > Creative Industries > Support for Artists",
    "37": "Culture > Cultural Access > Public Engagement",
    "38": "Infrastructure and Transport > Transport Modernization > Public Transport Investment",
    "39": "Infrastructure and Transport > Sustainable Transport > Electric Vehicles",
    "40": "Infrastructure and Transport > Sustainable Transport > Active Travel",
    "41": "Infrastructure and Transport > Urban Planning > Green Cities",
    "42": "Infrastructure and Transport > Rural Connectivity > Rural Transport Access",
    "43": "Employment and Workers' Rights > Workers' Rights > Workplace Protections",
    "44": "Employment and Workers' Rights > Minimum Wage > Living Wage Policies",
    "45": "Employment and Workers' Rights > Employment Equity > Gender Pay Gap",
    "46": "Employment and Workers' Rights > Workplace Conditions > Flexible Work",
    "47": "Employment and Workers' Rights > Workforce Development > Skills Training",
    "48": "Education Beyond Schools > Higher Education > Tuition Fees and Loans",
    "49": "Education Beyond Schools > Adult Education > Lifelong Learning",
    "50": "Education Beyond Schools > Apprenticeships > Vocational Training",
    "51": "Education Beyond Schools > Skills Development > STEM Education",
    "52": "Education Beyond Schools > Education Equity > Disadvantaged Students",
    "53": "Immigration and Border Policy > Immigration Reform > Legal Pathways",
    "54": "Immigration and Border Policy > Refugee Support > Asylum Policies",
    "55": "Immigration and Border Policy > Border Security > Immigration Control",
    "56": "Immigration and Border Policy > Integration and Community Support > Integration Support",
    "57": "Immigration and Border Policy > Post-Brexit Immigration Policies > Post-Brexit Policies",
    "58": "Justice and Law Reform > Criminal Justice Reform > Systemic Issues",
    "59": "Justice and Law Reform > Prison Reform > Rehabilitation Programs",
    "60": "Justice and Law Reform > Access to Legal Aid > Legal Support",
    "61": "Health Beyond the NHS > Social Care > Elderly and Disabled Support",
    "62": "Health Beyond the NHS > Public Health > Preventative Initiatives",
    "63": "Health Beyond the NHS > Health Inequalities > Equity in Care",
    "64": "Technology and Data Ethics > AI Ethics > Regulation",
    "65": "Technology and Data Ethics > Digital Rights > Privacy and Security",
    "66": "Technology and Data Ethics > Tech Accessibility > Inclusivity",
    "67": "Foreign Policy and International Relations > Global Leadership > Climate Diplomacy",
    "68": "Foreign Policy and International Relations > Trade Agreements > Post-Brexit Trade",
    "69": "Foreign Policy and International Relations > International Aid > Overseas Development",
    "70": "Taxation and Fiscal Responsibility > Corporate Tax > Corporate Accountability",
    "71": "Taxation and Fiscal Responsibility > Wealth Taxes > Progressive Taxation",
    "72": "Taxation and Fiscal Responsibility > Tax Reliefs > Small Business Support",
    "73": "Housing Beyond Welfare > Private Rental Market > Regulation",
    "74": "Housing Beyond Welfare > Homeownership > First-Time Buyers",
    "75": "Housing Beyond Welfare > Building Standards > Sustainable Housing",
    "76": "Climate Resilience > Flood Defenses > Infrastructure Investment",
    "77": "Climate Resilience > Natural Disasters > Preparedness",
    "78": "Climate Resilience > Biodiversity > Habitat Protection",
    "79": "Regional Development > Devolution > Local Governance",
    "80": "Regional Development > Levelling Up > Infrastructure Investment",
    "81": "Regional Development > Rural Policies > Community Support",
    "82": "Social Equality and Diversity > LGBTQ+ Rights > Protection and Advocacy",
    "83": "Social Equality and Diversity > Disability Rights > Accessibility",
    "84": "Social Equality and Diversity > Racial Equality > Diversity and Inclusion",
    "85": "National Identity and Union > Scottish Independence > Referendum Policies",
    "86": "National Identity and Union > Northern Ireland > Peace and Governance",
    "87": "National Identity and Union > Cultural Unity > UK Integration",
    "88": "Miscellaneous > General Questions > Manifesto Priorities",
    "89": "Miscellaneous > General Questions > Labour Values",
    "90": "Miscellaneous > General Questions > Labour Leadership",
    "91": "Miscellaneous > General Questions > Voter Engagement"
}


model = SetFitModel.from_pretrained("ckpt/", local_files_only=True)

input_text = "Thoughts on peace and governance"

              
output = model.predict(input_text)
output_label = int(output)

print(f"Predicted output class: {output_label}, which is intent: '{class_label_map[output_label]}'")