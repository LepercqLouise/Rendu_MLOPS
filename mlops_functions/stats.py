import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


# Définir une palette de couleurs
colors = ['#008030', '#008080', '#00A86B', '#4CAF50', '#7CFC00']

def plot_sex_pie_chart(data, title):
    plt.figure(figsize=(6, 6))
    colors = ['lightblue', 'lightcoral']
    data['Sex'].value_counts().plot.pie(autopct='%1.1f%%', shadow=True, startangle=15, colors=colors)
    plt.title(title)
    plt.show()

def plot_age_class_pie_chart(data, title):
    plt.figure(figsize=(6, 6))
    colors = ['#008030', '#008080', '#00A86B', '#4CAF50', '#7CFC00']
    data['Classe_age'].value_counts().plot.pie(autopct='%1.1f%%', shadow=True, startangle=15, colors=colors)
    plt.title(title)
    plt.show()

def plot_height_class_pie_chart(data, title):
    plt.figure(figsize=(6, 6))
    colors = ['#FF5733', '#FF8C00', '#FFD700', '#FFED00', '#FFFAF0']
    data['Classe_height'].value_counts().plot.pie(autopct='%1.1f%%', shadow=True, startangle=15, colors=colors)
    plt.title(title)
    plt.show()

def plot_count_percentage_bar_chart(data, column, title):
    plt.figure(figsize=(12, 6))
    total = float(len(data[column]))
    ax = sns.countplot(x=column, data=data, palette='viridis')

    def percent_formatter(x, pos):
        return f'{(x / total):.0%}'

    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height / 2,
                '{:.0%}'.format(height / total), ha="center")

    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Pourcentage')
    plt.xticks(rotation=45)
    plt.show()

def plot_noc_distribution(data):
    tableau_noc = data.groupby('NOC').size().reset_index(name='Nombre d\'individus')
    tableau_noc_trie = tableau_noc.sort_values(by='Nombre d\'individus', ascending=False)
    les_16_premiers = tableau_noc_trie.head(16)

    # Afficher les 16 premiers pays les plus représentés
    print(les_16_premiers)

