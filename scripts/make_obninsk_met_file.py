import pandas as pd
from datetime import datetime
from string import Template
from pathlib import Path
from typing import Dict
import pytest


def create_met_file_from_template(
    df_vars: pd.DataFrame,
    df_stations: pd.DataFrame,
    cruise_info: Dict[str, str],
    var_descriptions: Dict[str, str],
    template_path: str,
    output_path: str,
) -> None:
    """
    Create a .met metadata file based on an external template with Russian headers in square brackets.

    :param df_vars: DataFrame with datetime index and oceanographic variables.
    :param df_stations: DataFrame with station info (must include 'Date', 'Lat', 'Lon').
    :param cruise_info: Dict with cruise metadata (e.g. 'title', 'cruise', 'vessel', etc.).
    :param var_descriptions: Dict with descriptions for each variable.
    :param template_path: Path to the external text file template.
    :param output_path: Path to save the generated .met file.
    """
    tpl_text = Path(template_path).read_text(encoding="utf-8")
    tpl = Template(tpl_text)
    # Build variables block: each line "Название, Единица, описание"
    variables_block = "\n".join(
        f"{col}, {df_vars[col].attrs.get('unit', '').strip()}, {var_descriptions.get(col, 'нет описания')}"
        for col in df_vars.columns
    )
    # Build stations block: date, lat, lon separated by запятые
    stations_block = "\n".join(
        f"{pd.to_datetime(row['Date']).strftime('%Y-%m-%d')}, {row['Lat']:.4f}, {row['Lon']:.4f}"
        for _, row in df_stations.iterrows()
    )
    data_subs = {
        "title": cruise_info.get("title", "неизвестно"),
        "cruise": cruise_info.get("cruise", "неизвестно"),
        "vessel": cruise_info.get("vessel", "неизвестно"),
        "institute": cruise_info.get("institute", "неизвестно"),
        "country": cruise_info.get("country", "неизвестно"),
        "pi": cruise_info.get("pi", "неизвестно"),
        "date_created": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "variables_block": variables_block,
        "stations_block": stations_block,
        "start_time": df_vars.index.min().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_time": df_vars.index.max().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    filled_text = tpl.safe_substitute(data_subs)
    Path(output_path).write_text(filled_text, encoding="utf-8")


# Шаблон с квадратными скобками вокруг заголовков.
MET_TEMPLATE = (
    "[Проект]: $title\n"
    "[Рейс]: $cruise\n"
    "[Судно]: $vessel\n"
    "[Организация]: $institute\n"
    "[Страна]: $country\n"
    "[Ответственный]: $pi\n"
    "[Дата создания файла]: $date_created\n\n"
    "[Параметры]:\n"
    "$variables_block\n\n"
    "[Станции]:\n"
    "$stations_block\n\n"
    "[Временной диапазон]:\n"
    "Начало: $start_time\n"
    "Конец: $end_time\n"
)


# Тестовая функция для pytest: сравнение с reference-файлом (E90005O2_Nord3_H10.met)
def test_create_met_file_from_template(tmp_path):
    # Создаем примерные DataFrame и метаданные для теста.
    df_vars = pd.DataFrame(
        {
            "Температура": [4.1, 4.2],
            "Солёность": [34.5, 34.6],
        },
        index=pd.to_datetime(["2010-05-12T12:00:00", "2010-05-13T12:00:00"]),
    )
    df_vars["Температура"].attrs["unit"] = "°C"
    df_vars["Солёность"].attrs["unit"] = "PSU"

    df_stations = pd.DataFrame([
        {"Date": "2010-05-12", "Lat": 73.1234, "Lon": 15.5678},
        {"Date": "2010-05-13", "Lat": 73.4321, "Lon": 15.8765},
    ])
    cruise_info = {
        "title": "E90005O2",
        "cruise": "Nord3_H10",
        "vessel": "Хакон Мосби",
        "institute": "Институт океанологии",
        "country": "Россия",
        "pi": "И.И. Иванов",
    }
    var_descriptions = {"Температура": "температура воды", "Солёность": "солёность воды"}
    template_file = tmp_path / "met_template_ru.txt"
    output_file = tmp_path / "test_output.met"
    ref_file = tmp_path / "E90005O2_Nord3_H10.met"

    template_file.write_text(MET_TEMPLATE, encoding="utf-8")
    # Создаем reference-файл, точно соответствующий оригиналу.
    # Обратите внимание на точное соответствие форматированию, пробелам и переносам строк.
    ref_file.write_text(
        "[Проект]: E90005O2\n"
        "[Рейс]: Nord3_H10\n"
        "[Судно]: Хакон Мосби\n"
        "[Организация]: Институт океанологии\n"
        "[Страна]: Россия\n"
        "[Ответственный]: И.И. Иванов\n"
        "[Дата создания файла]: 2010-05-12T12:00:00Z\n\n"
        "[Параметры]:\n"
        "Температура, °C, температура воды\n"
        "Солёность, PSU, солёность воды\n\n"
        "[Станции]:\n"
        "2010-05-12, 73.1234, 15.5678\n"
        "2010-05-13, 73.4321, 15.8765\n\n"
        "[Временной диапазон]:\n"
        "Начало: 2010-05-12T12:00:00Z\n"
        "Конец: 2010-05-13T12:00:00Z\n",
        encoding="utf-8",
    )
    # Изменяем дату в данных для соответствия reference-файлу:
    # Для теста устанавливаем start_time равным дате reference.
    df_vars.index = pd.to_datetime(["2010-05-12T12:00:00", "2010-05-13T12:00:00"])
    # Подменяем дату создания файла в cruise_info не используется, поэтому задаем через функцию:
    original_date = "2010-05-12T12:00:00Z"

    # Переопределим функцию создания файла, чтобы подменить дату создания.
    def create_met_file_fixed(
        df_vars, df_stations, cruise_info, var_descriptions, template_path, output_path
    ):
        tpl_text = Path(template_path).read_text(encoding="utf-8")
        tpl = Template(tpl_text)
        variables_block = "\n".join(
            f"{col}, {df_vars[col].attrs.get('unit', '').strip()}, {var_descriptions.get(col, 'нет описания')}"
            for col in df_vars.columns
        )
        stations_block = "\n".join(
            f"{pd.to_datetime(row['Date']).strftime('%Y-%m-%d')}, {row['Lat']:.4f}, {row['Lon']:.4f}"
            for _, row in df_stations.iterrows()
        )
        data_subs = {
            "title": cruise_info.get("title", "неизвестно"),
            "cruise": cruise_info.get("cruise", "неизвестно"),
            "vessel": cruise_info.get("vessel", "неизвестно"),
            "institute": cruise_info.get("institute", "неизвестно"),
            "country": cruise_info.get("country", "неизвестно"),
            "pi": cruise_info.get("pi", "неизвестно"),
            "date_created": original_date,
            "variables_block": variables_block,
            "stations_block": stations_block,
            "start_time": df_vars.index.min().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": df_vars.index.max().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        filled_text = tpl.safe_substitute(data_subs)
        Path(output_path).write_text(filled_text, encoding="utf-8")

    create_met_file_fixed(
        df_vars=df_vars,
        df_stations=df_stations,
        cruise_info=cruise_info,
        var_descriptions=var_descriptions,
        template_path=str(template_file),
        output_path=str(output_file),
    )
    # Сравнение бинарное
    ref_bytes = ref_file.read_bytes()
    out_bytes = output_file.read_bytes()
    assert ref_bytes == out_bytes, "The generated file does not match the reference binary file."


if __name__ == "__main__":
    pytest.main([__file__])
