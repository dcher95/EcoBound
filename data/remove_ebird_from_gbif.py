import pandas as pd

def main():
    # === CONFIGURATION ===
    gbif_csv = "/data/cher/EcoBound/data/gbif_full_filtered.csv"
    ebird_csv = "/data/cher/EcoBound/data/eBird/eBird_STL.csv"
    output_csv = "/data/cher/EcoBound/data/gbif_full_no_overlap.csv"
    inat_csv = "/data/cher/EcoBound/data/inat_full.csv"
    
    # Load the iNat and eBird datasets
    gbif_csv = pd.read_csv(gbif_csv)

    ## For removing eBird observations from GBIF
    # ebird_df = pd.read_csv(ebird_csv)
    
    # # Filter out observations made by the same observer on the same date in the eBird dataset from the iNat dataset
    # # We assume both datasets have 'observation_date' and 'observer_id'
    # merged_df = gbif_csv.merge(
    #     ebird_df[['observation_date', 'observer_id']],
    #     left_on=['eventDate', 'recordedBy'],
    #     right_on=['observation_date', 'observer_id'],
    #     how='left',
    #     indicator=True
    # )
    # filtered_gbif_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    
    # Save the filtered GBIF dataset to a new CSV file
    # filtered_gbif_df.to_csv(output_csv, index=False)
    # print(f"Filtered GBIF dataset saved to {output_csv}")

    # Save the filtered iNat dataset to a new CSV file
    filtered_inat_df = gbif_csv[gbif_csv['occurrenceID'].str.contains('inaturalist', na=False)]
    import code; code.interact(local=locals())
    filtered_inat_df.to_csv(inat_csv, index=False)
    print(f"Filtered GBIF dataset saved to {output_csv}")

if __name__ == '__main__':
    main()