#!/usr/bin/env python3
"""
Verify that the correct reading types are in the correct split files.
"""

import pandas as pd

def verify_split_files():
    """
    Verify that MT, CT, and Pn data went to the correct files.
    """
    print("🔍 Verifying split files have correct reading types...")
    
    # Load the original filtered data
    print("📂 Loading original filtered data...")
    original = pd.read_csv('../../data/processed/dam_data_filtered.csv')
    original['datetime'] = pd.to_datetime(original['datetime'])
    
    # Load the split files
    print("📂 Loading split files...")
    mt_file = pd.read_csv('../../data/processed/dam_data_MT.csv')
    ct_file = pd.read_csv('../../data/processed/dam_data_CT.csv')
    pn_file = pd.read_csv('../../data/processed/dam_data_Pn.csv')
    
    mt_file['datetime'] = pd.to_datetime(mt_file['datetime'])
    ct_file['datetime'] = pd.to_datetime(ct_file['datetime'])
    pn_file['datetime'] = pd.to_datetime(pn_file['datetime'])
    
    print(f"   Original: {len(original):,} rows")
    print(f"   MT file: {len(mt_file):,} rows")
    print(f"   CT file: {len(ct_file):,} rows")
    print(f"   Pn file: {len(pn_file):,} rows")
    
    # Get original data by reading type
    original_mt = original[original['reading'] == 'MT'].copy()
    original_ct = original[original['reading'] == 'CT'].copy()
    original_pn = original[original['reading'] == 'Pn'].copy()
    
    print(f"\n📊 Original data by reading type:")
    print(f"   MT: {len(original_mt):,} rows")
    print(f"   CT: {len(original_ct):,} rows")
    print(f"   Pn: {len(original_pn):,} rows")
    
    # Compare values for the same timestamp/fly combination
    print(f"\n🔍 Comparing values for same timestamp/fly...")
    
    # Pick a specific timestamp and fly to compare
    test_timestamp = original['datetime'].iloc[0]
    test_fly = original['fly_id'].iloc[0]
    
    print(f"   Test timestamp: {test_timestamp}")
    print(f"   Test fly: {test_fly}")
    
    # Get original values
    orig_mt_val = original_mt[(original_mt['datetime'] == test_timestamp) & 
                             (original_mt['fly_id'] == test_fly)]['value'].iloc[0]
    orig_ct_val = original_ct[(original_ct['datetime'] == test_timestamp) & 
                             (original_ct['fly_id'] == test_fly)]['value'].iloc[0]
    orig_pn_val = original_pn[(original_pn['datetime'] == test_timestamp) & 
                             (original_pn['fly_id'] == test_fly)]['value'].iloc[0]
    
    # Get split file values
    split_mt_val = mt_file[(mt_file['datetime'] == test_timestamp) & 
                          (mt_file['fly_id'] == test_fly)]['value'].iloc[0]
    split_ct_val = ct_file[(ct_file['datetime'] == test_timestamp) & 
                          (ct_file['fly_id'] == test_fly)]['value'].iloc[0]
    split_pn_val = pn_file[(pn_file['datetime'] == test_timestamp) & 
                          (pn_file['fly_id'] == test_fly)]['value'].iloc[0]
    
    print(f"\n   Original values:")
    print(f"     MT: {orig_mt_val}")
    print(f"     CT: {orig_ct_val}")
    print(f"     Pn: {orig_pn_val}")
    
    print(f"\n   Split file values:")
    print(f"     MT file: {split_mt_val}")
    print(f"     CT file: {split_ct_val}")
    print(f"     Pn file: {split_pn_val}")
    
    # Verify matches
    mt_match = orig_mt_val == split_mt_val
    ct_match = orig_ct_val == split_ct_val
    pn_match = orig_pn_val == split_pn_val
    
    print(f"\n   Verification:")
    print(f"     MT file has MT values: {'✅' if mt_match else '❌'}")
    print(f"     CT file has CT values: {'✅' if ct_match else '❌'}")
    print(f"     Pn file has Pn values: {'✅' if pn_match else '❌'}")
    
    # Check a few more random samples
    print(f"\n🔍 Checking 5 random samples...")
    
    for i in range(5):
        sample_idx = i * 1000  # Every 1000th row
        if sample_idx < len(original):
            sample_timestamp = original['datetime'].iloc[sample_idx]
            sample_fly = original['fly_id'].iloc[sample_idx]
            
            # Get original values
            orig_mt = original_mt[(original_mt['datetime'] == sample_timestamp) & 
                                 (original_mt['fly_id'] == sample_fly)]['value']
            orig_ct = original_ct[(original_ct['datetime'] == sample_timestamp) & 
                                 (original_ct['fly_id'] == sample_fly)]['value']
            orig_pn = original_pn[(original_pn['datetime'] == sample_timestamp) & 
                                 (original_pn['fly_id'] == sample_fly)]['value']
            
            if len(orig_mt) > 0 and len(orig_ct) > 0 and len(orig_pn) > 0:
                # Get split file values
                split_mt = mt_file[(mt_file['datetime'] == sample_timestamp) & 
                                  (mt_file['fly_id'] == sample_fly)]['value']
                split_ct = ct_file[(ct_file['datetime'] == sample_timestamp) & 
                                  (ct_file['fly_id'] == sample_fly)]['value']
                split_pn = pn_file[(pn_file['datetime'] == sample_timestamp) & 
                                  (pn_file['fly_id'] == sample_fly)]['value']
                
                if len(split_mt) > 0 and len(split_ct) > 0 and len(split_pn) > 0:
                    mt_ok = orig_mt.iloc[0] == split_mt.iloc[0]
                    ct_ok = orig_ct.iloc[0] == split_ct.iloc[0]
                    pn_ok = orig_pn.iloc[0] == split_pn.iloc[0]
                    
                    print(f"   Sample {i+1}: MT={mt_ok}, CT={ct_ok}, Pn={pn_ok}")
    
    # Final verification
    print(f"\n" + "=" * 50)
    print("=== VERIFICATION SUMMARY ===")
    print("=" * 50)
    
    if mt_match and ct_match and pn_match:
        print("✅ ALL FILES HAVE CORRECT READING TYPES!")
        print("   - MT file contains MT (movement) values")
        print("   - CT file contains CT (cumulative) values") 
        print("   - Pn file contains Pn (pause) values")
    else:
        print("❌ ERROR: Files have incorrect reading types!")
        print("   Please check the splitting logic.")

if __name__ == "__main__":
    verify_split_files()
