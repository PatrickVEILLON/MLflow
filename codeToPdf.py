import pdfkit

pdfkit_config = pdfkit.configuration(wkhtmltopdf='C:/chemin/vers/wkhtmltopdf')

# Lire le contenu HTML depuis le fichier
with open('Weather-Dataset-Logistic-Regression _ Kaggle (21_02_2024 13_43_47).html', 'r', encoding='utf-8') as file:
    html_content = file.read()

# Analyser le contenu HTML
soup = BeautifulSoup(html_content, 'html.parser')

# Générer le PDF
pdfkit.from_string(str(soup), 'output.pdf', configuration=pdfkit_config)
