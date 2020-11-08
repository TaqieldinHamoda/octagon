import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook

# Global Variables and refrences to data. DO NOT MODIFY
EXCEL = "./Data.xlsx"
SEX = ('ALL', 'M', 'F')
PROV = ('ALL', 'AB', 'ON', 'QC', 'BC', 'Atlantic', 'Prairies', 'UNKNWN')
CON_ACT = ('ALL', 'Yes', 'No', 'Null')
MEASURE = ('Tx', 'events', 'censored')
AGES = tuple(["ALL", "18-19"] + [f"{i}-{i+4}" for i in range(20, 61, 5)] + ["65+"])


def get_data():
    """
    :return: A pandas data frame with all of the data
    """
    turn_int = lambda x: 0 if x is None else int(x)
    sheet = load_workbook(filename=EXCEL, read_only=True, data_only=True).active
    legend = []
    data = []

    for row in sheet.iter_rows(values_only=True):
        legend.append(row[:5])
        data.append(tuple(map(turn_int, row[5:])))

    legend = pd.DataFrame(np.array(legend), columns=("prov", "con_act", "sex", "age", "measure"))
    data = pd.DataFrame(np.array(data), columns=[f"M{i}" for i in range(40)])

    return pd.concat([legend, data], axis=1, sort=False)


def filter_data(prov=PROV, con_act=CON_ACT, sex=SEX, ages=AGES, measure=MEASURE, m_start=0, m_end=39):
    """
    :param prov: A list with all provinces that you want to get data for
    :param con_act: A list with all cancer statuses that you want to get data for
    :param sex: A list with all sex that you want to get data for
    :param ages: A list with all ages that you want to get data for
    :param measure: A list with all measures that you want to get data for
    :param m_start: The starting months for data. Default is maximum which is 0
    :param m_end: The ending months for data. Default is maximum which is 39
    :return: filtered data as pandas dataframe
    """
    data = get_data()
    data = data[data.prov.isin(prov) & data.con_act.isin(con_act) & data.sex.isin(sex)
                & data.age.isin(ages) & data.measure.isin(measure)]

    return pd.concat([data.iloc[:, :5], data.iloc[:, m_start + 5:m_end + 6]], axis=1, sort=False)


def plot_province(prov="ALL", measure="Tx", age="ALL", m_start=0, m_end=39):
    """
    :param prov: String of the province to plot data for
    :param measure: String of the measure to plot data for
    :param age: String of the age to plot data for
    :param m_start: Integer of the start month
    :param m_end: Integer of the end month
    :return: VOID. Just plots the data
    """

    # Initialize the data and filter it
    data = filter_data(prov=[prov], measure=[measure], ages=[age], m_start=m_start, m_end=m_end)
    data_female = data[data.sex == "F"]
    data_male = data[data.sex == "M"]

    female_total = [0 for i in range(len(CON_ACT))]
    male_total = [0 for i in range(len(CON_ACT))]

    # Get all the sums for the data
    for i in range(len(CON_ACT)):
        if data_female[data_female.con_act == CON_ACT[i]].shape[0] > 0:
            female_total[i] += np.sum(data_female[data_female.con_act == CON_ACT[i]].iloc[0, 5:].values)
        if data_male[data_male.con_act == CON_ACT[i]].shape[0] > 0:
            male_total[i] += np.sum(data_male[data_male.con_act == CON_ACT[i]].iloc[0, 5:].values)

    x = np.arange(len(CON_ACT))  # the label locations
    width = 0.35  # the width of the bars

    # Create the rectangles
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, male_total, width, label='Male')
    rects2 = ax.bar(x + width / 2, female_total, width, label='Female')

    # Place the rectangles in the right place
    for rects in (rects1, rects2):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x)
    ax.set_xticklabels(CON_ACT)
    ax.legend()
    ax.set_ylabel(f"Number of participants from month {m_start} to month {m_end}")
    ax.set_xlabel("Is the patient using another anti-cancer treatment?")

    if "ALL" == age:
        if "ALL" == prov:
            ax.set_title(f"National data for ages 18+")
        else:
            ax.set_title(f"Data for {prov} for ages 18+")
    else:
        if "ALL" == prov:
            ax.set_title(f"National data for ages {age}")
        else:
            ax.set_title(f"Data for {prov} for ages {age}")

    fig.tight_layout()
    plt.show()


def discontinuation():
    pass


if __name__ == '__main__':
    # Question 1 answer
    plot_province(m_start=9)
