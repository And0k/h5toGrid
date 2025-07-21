import copernicus_marine_client as cm

points = (
    (55.00, 13.75),
    (55.437, 14.87),
    (55.24, 14.87),
    (55.30, 15.30),
    (55.2, 15.26)
)
dates = ('2023-12-15', '2024-01-31', '2024-02-01')
dataset_vars = {
    'cmems_mod_bal_bgc_anfc_P1D-m': ['o2b'],
    'cmems_mod_bal_phy_anfc_P1D-m': ('bottomT', 'sob')
}
results = []
for dataset, vars in dataset_vars.items():
    for date in dates:
        print(date)
        for lat, lon in points:
            df = cm.read_dataframe(
                dataset_id = dataset,
                minimum_longitude = lon,
                maximum_longitude = lon,
                minimum_latitude = lat,
                maximum_latitude = lat,
                variables = vars,
                start_datetime = date,
                end_datetime = date
            )
            results.append(df)
d = pd.concat(results, sort=True)
d
d.groupby(level=[0,1,2]).first()
d.o2b_ppm = d.o2b*0.0319988
d['o2b_ppm'] = d.o2b*0.0319988
dd = d.groupby(level=[0,1,2]).first()[['bottomT', 'sob', 'o2b', 'o2b_ppm']]
dd.reorder_levels([1,2,0]).sort_index()



if False:
    dataset_vars = {
        'cmems_mod_bal_bgc_anfc_P1D-m': "o2,o2b".split(','),
        'cmems_mod_bal_phy_anfc_P1D-m': ('uo', 'vo', 'wo', 'so', 'thetao', 'bottomT', 'sob')
    }
    for dataset, vars in dataset_vars.items():
        print(dataset, end=': ')
        cm.subset(
            dataset_id=dataset,
            variables=vars,
            minimum_longitude=10.5,
            maximum_longitude=21.33,     # 20 17.5
            minimum_latitude=54.0,
            maximum_latitude=59,    # 56.5 56.3
            start_datetime="2023-12-01",
            end_datetime="2024-02-10",
            minimum_depth=0.5,
            maximum_depth=100,
            output_directory = r"d:\workData\BalticSea\_other_data\_model\NEMO@CMEMS\section_z"
        )
