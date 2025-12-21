#!/usr/bin/env python3
"""
Comprehensive test suite for 2-remove_flies.py

This test suite covers:
- Individual removal criteria (status, flies, genotypes, sexes, treatments, per-fly days)
- Multiple criteria combinations
- Multiple values within criteria
- Edge cases (empty lists, non-existent values, missing data, etc.)

How to run:
    # From the test-scripts directory:
    pytest test_remove_flies.py -v
    
    # Or run directly with Python:
    python test_remove_flies.py
    
    # Run specific test class:
    pytest test_remove_flies.py::TestIndividualCriteria -v
    
    # Run specific test:
    pytest test_remove_flies.py::TestIndividualCriteria::test_remove_by_status_dead -v

    # -v is the verbose/descriptive version
Requirements:
    pip install pytest pandas numpy
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the pipeline directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent))

# Import functions from the module (handle module name starting with number)
import importlib.util
spec = importlib.util.spec_from_file_location("remove_flies_module", 
    Path(__file__).parent / "../pipeline/2-remove_flies.py")
remove_flies_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remove_flies_module)

# Import the functions we need
remove_flies = remove_flies_module.remove_flies
create_fly_key = remove_flies_module.create_fly_key


# ============================================================
#   TEST DATA GENERATION
# ============================================================

def create_test_data():
    """Create a comprehensive test dataset."""
    np.random.seed(42)  # For reproducibility
    
    # Create data for 10 flies across 2 monitors, 5 days, 24 hours per day
    monitors = [5, 6]
    channels = [1, 2, 3, 4, 5]
    days = [1, 2, 3, 4, 5]
    hours = list(range(24))
    
    genotypes = ['SSS', 'Rye', 'Fmn', 'Iso']
    sexes = ['Male', 'Female']
    treatments = ['2mM His', '8mM His', 'VEH']
    
    rows = []
    for monitor in monitors:
        for channel in channels:
            fly_num = (monitor - 5) * 5 + channel
            genotype = genotypes[fly_num % len(genotypes)]
            sex = sexes[fly_num % len(sexes)]
            treatment = treatments[fly_num % len(treatments)]
            
            for day in days:
                for hour in hours:
                    rows.append({
                        'monitor': monitor,
                        'channel': channel,
                        'datetime': pd.Timestamp(f'2025-01-{day:02d} {hour:02d}:00:00'),
                        'value': np.random.randint(0, 10),
                        'reading': 'MT',
                        'genotype': genotype,
                        'sex': sex,
                        'treatment': treatment,
                        'Exp_Day': day,
                        'ZT': hour,
                        'Phase': 'Light' if 6 <= hour < 18 else 'Dark'
                    })
    
    df = pd.DataFrame(rows)
    return df


def create_test_health_report():
    """Create a test health report with various statuses."""
    data = {
        'Monitor': [5, 5, 5, 5, 5, 6, 6, 6, 6, 6],
        'Channel': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'Genotype': ['SSS', 'Rye', 'Fmn', 'Iso', 'SSS', 'Rye', 'Fmn', 'Iso', 'SSS', 'Rye'],
        'Sex': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Treatment': ['2mM His', '8mM His', 'VEH', '2mM His', '8mM His', 'VEH', '2mM His', '8mM His', 'VEH', '2mM His'],
        'FINAL_STATUS': ['Alive', 'Dead', 'Unhealthy', 'Alive', 'Dead', 'Alive', 'Unhealthy', 'Alive', 'QC_Fail', 'Alive'],
        'DAYS_ALIVE': [5, 2, 3, 5, 1, 5, 4, 5, 0, 5],
        'DAYS_DEAD': [0, 3, 0, 0, 4, 0, 0, 0, 5, 0],
        'DAYS_UNHEALTHY': [0, 0, 2, 0, 0, 0, 1, 0, 0, 0]
    }
    return pd.DataFrame(data)


# ============================================================
#   FIXTURES
# ============================================================

@pytest.fixture
def test_data():
    """Fixture providing test data."""
    return create_test_data()


@pytest.fixture
def health_report():
    """Fixture providing test health report."""
    return create_test_health_report()


# ============================================================
#   TESTS: Individual Removal Criteria
# ============================================================

class TestIndividualCriteria:
    """Test each removal criterion individually."""
    
    def test_remove_by_status_dead(self, test_data, health_report):
        """Test removing flies with 'Dead' status."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            statuses_to_remove=['Dead'],
            health_report=health_report
        )
        
        # Check that dead flies are removed (Monitor 5 Channel 2 and 5)
        # According to health report: 5-ch2=Dead, 5-ch5=Dead (2 flies total)
        dead_fly_keys = ['5-ch2', '5-ch5']
        remaining_fly_keys = result_df.apply(
            lambda row: create_fly_key(row['monitor'], row['channel']), axis=1
        ).unique()
        
        for fly_key in dead_fly_keys:
            assert fly_key.lower() not in [k.lower() for k in remaining_fly_keys]
        
        assert summary['rows_removed_by_status'] > 0
        assert summary['flies_removed'] == 2  # 2 dead flies (5-ch2, 5-ch5)
    
    def test_remove_by_status_unhealthy(self, test_data, health_report):
        """Test removing flies with 'Unhealthy' status."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            statuses_to_remove=['Unhealthy'],
            health_report=health_report
        )
        
        # Unhealthy flies: Monitor 5 Channel 3, Monitor 6 Channel 2
        unhealthy_fly_keys = ['5-ch3', '6-ch2']
        remaining_fly_keys = result_df.apply(
            lambda row: create_fly_key(row['monitor'], row['channel']), axis=1
        ).unique()
        
        for fly_key in unhealthy_fly_keys:
            assert fly_key.lower() not in [k.lower() for k in remaining_fly_keys]
        
        assert summary['rows_removed_by_status'] > 0
        assert summary['flies_removed'] == 2
    
    def test_remove_by_status_multiple(self, test_data, health_report):
        """Test removing flies with multiple statuses."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            statuses_to_remove=['Dead', 'Unhealthy', 'QC_Fail'],
            health_report=health_report
        )
        
        # Should remove: Dead (2: 5-ch2, 5-ch5), Unhealthy (2: 5-ch3, 6-ch2), QC_Fail (1: 6-ch4) = 5 flies
        assert summary['flies_removed'] == 5
        assert summary['rows_removed_by_status'] > 0
    
    def test_remove_by_specific_flies(self, test_data):
        """Test removing specific flies by ID."""
        flies_to_remove = ['5-ch1', '5-ch3', '6-ch2']
        result_df, summary, total_removed = remove_flies(
            test_data,
            flies_to_remove=flies_to_remove
        )
        
        remaining_fly_keys = result_df.apply(
            lambda row: create_fly_key(row['monitor'], row['channel']), axis=1
        ).unique()
        
        for fly_key in flies_to_remove:
            assert fly_key.lower() not in [k.lower() for k in remaining_fly_keys]
        
        assert summary['rows_removed_by_fly'] > 0
        assert summary['flies_removed'] == 3
    
    def test_remove_by_genotype_single(self, test_data):
        """Test removing flies by single genotype."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            genotypes_to_remove=['SSS']
        )
        
        # Check that no SSS flies remain
        if 'genotype' in result_df.columns:
            assert 'SSS' not in result_df['genotype'].values
        
        assert summary['rows_removed_by_metadata'] > 0
    
    def test_remove_by_genotype_multiple(self, test_data):
        """Test removing flies by multiple genotypes."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            genotypes_to_remove=['SSS', 'Rye']
        )
        
        # Check that no SSS or Rye flies remain
        if 'genotype' in result_df.columns:
            remaining_genotypes = result_df['genotype'].unique()
            assert 'SSS' not in remaining_genotypes
            assert 'Rye' not in remaining_genotypes
        
        assert summary['rows_removed_by_metadata'] > 0
    
    def test_remove_by_sex_single(self, test_data):
        """Test removing flies by single sex."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            sexes_to_remove=['Female']
        )
        
        # Check that no Female flies remain
        if 'sex' in result_df.columns:
            assert 'Female' not in result_df['sex'].values
        
        assert summary['rows_removed_by_metadata'] > 0
    
    def test_remove_by_sex_multiple(self, test_data):
        """Test removing flies by multiple sexes (should remove all)."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            sexes_to_remove=['Male', 'Female']
        )
        
        # Should remove all flies
        assert len(result_df) == 0
        assert summary['rows_removed_by_metadata'] == len(test_data)
    
    def test_remove_by_treatment_single(self, test_data):
        """Test removing flies by single treatment."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            treatments_to_remove=['VEH']
        )
        
        # Check that no VEH flies remain
        if 'treatment' in result_df.columns:
            assert 'VEH' not in result_df['treatment'].values
        
        assert summary['rows_removed_by_metadata'] > 0
    
    def test_remove_by_treatment_multiple(self, test_data):
        """Test removing flies by multiple treatments."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            treatments_to_remove=['2mM His', '8mM His']
        )
        
        # Check that only VEH flies remain
        if 'treatment' in result_df.columns:
            remaining_treatments = result_df['treatment'].unique()
            assert 'VEH' in remaining_treatments or len(remaining_treatments) == 0
            assert '2mM His' not in remaining_treatments
            assert '8mM His' not in remaining_treatments
        
        assert summary['rows_removed_by_metadata'] > 0
    
    def test_remove_by_per_fly_days_single(self, test_data):
        """Test removing specific days for a single fly."""
        per_fly_remove = {'5-ch1': [1, 2]}
        result_df, summary, total_removed = remove_flies(
            test_data,
            per_fly_remove=per_fly_remove
        )
        
        # Check that days 1 and 2 are removed for fly 5-ch1
        fly_data = result_df[
            (result_df['monitor'] == 5) & (result_df['channel'] == 1)
        ]
        if len(fly_data) > 0:
            assert 1 not in fly_data['Exp_Day'].values
            assert 2 not in fly_data['Exp_Day'].values
            # Days 3, 4, 5 should still be there
            assert 3 in fly_data['Exp_Day'].values or len(fly_data) == 0
        
        assert summary['rows_removed_by_days'] > 0
    
    def test_remove_by_per_fly_days_multiple(self, test_data):
        """Test removing specific days for multiple flies."""
        per_fly_remove = {
            '5-ch1': [1, 2],
            '5-ch2': [3, 4],
            '6-ch1': [5]
        }
        result_df, summary, total_removed = remove_flies(
            test_data,
            per_fly_remove=per_fly_remove
        )
        
        assert summary['rows_removed_by_days'] > 0
        
        # Verify removals for each fly
        for fly_key, days in per_fly_remove.items():
            monitor, channel = fly_key.split('-ch')
            monitor = int(monitor)
            channel = int(channel)
            fly_data = result_df[
                (result_df['monitor'] == monitor) & (result_df['channel'] == channel)
            ]
            if len(fly_data) > 0:
                for day in days:
                    assert day not in fly_data['Exp_Day'].values


# ============================================================
#   TESTS: Multiple Criteria Combinations
# ============================================================

class TestMultipleCriteria:
    """Test combinations of multiple removal criteria."""
    
    def test_status_and_treatment(self, test_data, health_report):
        """Test removing by status AND treatment."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            statuses_to_remove=['Dead'],
            treatments_to_remove=['VEH'],
            health_report=health_report
        )
        
        # Should remove both dead flies AND VEH flies
        assert summary['rows_removed_by_status'] > 0
        assert summary['rows_removed_by_metadata'] > 0
        assert total_removed > 0
    
    def test_status_and_genotype(self, test_data, health_report):
        """Test removing by status AND genotype."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            statuses_to_remove=['Dead', 'Unhealthy'],
            genotypes_to_remove=['SSS'],
            health_report=health_report
        )
        
        assert summary['rows_removed_by_status'] > 0
        assert summary['rows_removed_by_metadata'] > 0
    
    def test_sex_and_treatment(self, test_data):
        """Test removing by sex AND treatment."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            sexes_to_remove=['Female'],
            treatments_to_remove=['8mM His']
        )
        
        # Should remove Female flies AND 8mM His flies
        assert summary['rows_removed_by_metadata'] > 0
        
        # Verify removals
        if len(result_df) > 0:
            if 'sex' in result_df.columns:
                assert 'Female' not in result_df['sex'].values
            if 'treatment' in result_df.columns:
                assert '8mM His' not in result_df['treatment'].values
    
    def test_genotype_and_treatment_and_sex(self, test_data):
        """Test removing by genotype AND treatment AND sex."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            genotypes_to_remove=['Fmn'],
            treatments_to_remove=['VEH'],
            sexes_to_remove=['Male']
        )
        
        assert summary['rows_removed_by_metadata'] > 0
        
        # Verify all three criteria are applied
        if len(result_df) > 0:
            if 'genotype' in result_df.columns:
                assert 'Fmn' not in result_df['genotype'].values
            if 'treatment' in result_df.columns:
                assert 'VEH' not in result_df['treatment'].values
            if 'sex' in result_df.columns:
                assert 'Male' not in result_df['sex'].values
    
    def test_flies_and_status(self, test_data, health_report):
        """Test removing specific flies AND by status."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            flies_to_remove=['5-ch1', '6-ch5'],
            statuses_to_remove=['Dead'],
            health_report=health_report
        )
        
        assert summary['rows_removed_by_fly'] > 0
        assert summary['rows_removed_by_status'] > 0
    
    def test_per_fly_days_and_treatment(self, test_data):
        """Test removing per-fly days AND treatment."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            per_fly_remove={'5-ch1': [1, 2]},
            treatments_to_remove=['VEH']
        )
        
        assert summary['rows_removed_by_days'] > 0
        assert summary['rows_removed_by_metadata'] > 0
    
    def test_all_criteria_combined(self, test_data, health_report):
        """Test combining all removal criteria."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            flies_to_remove=['5-ch1'],
            per_fly_remove={'5-ch2': [1]},
            genotypes_to_remove=['Iso'],
            sexes_to_remove=['Female'],
            treatments_to_remove=['VEH'],
            statuses_to_remove=['Dead'],
            health_report=health_report
        )
        
        # All criteria should have some effect
        assert total_removed > 0


# ============================================================
#   TESTS: Edge Cases
# ============================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_no_criteria_specified(self, test_data):
        """Test with no removal criteria (should return original data)."""
        result_df, summary, total_removed = remove_flies(test_data)
        
        assert len(result_df) == len(test_data)
        assert total_removed == 0
        assert summary['flies_removed'] == 0
    
    def test_empty_lists(self, test_data):
        """Test with empty removal lists."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            flies_to_remove=[],
            genotypes_to_remove=[],
            sexes_to_remove=[],
            treatments_to_remove=[],
            statuses_to_remove=[]
        )
        
        assert len(result_df) == len(test_data)
        assert total_removed == 0
    
    def test_none_values(self, test_data):
        """Test with None values (should be same as empty)."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            flies_to_remove=None,
            genotypes_to_remove=None,
            sexes_to_remove=None,
            treatments_to_remove=None,
            statuses_to_remove=None
        )
        
        assert len(result_df) == len(test_data)
    
    def test_non_existent_flies(self, test_data):
        """Test removing flies that don't exist."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            flies_to_remove=['999-ch999', '888-ch888']
        )
        
        # Should not crash, but no removals
        assert len(result_df) == len(test_data)
        assert summary['rows_removed_by_fly'] == 0
    
    def test_non_existent_genotype(self, test_data):
        """Test removing non-existent genotype."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            genotypes_to_remove=['NonExistentGenotype']
        )
        
        # Should not crash, but no removals
        assert len(result_df) == len(test_data)
        assert summary['rows_removed_by_metadata'] == 0
    
    def test_non_existent_treatment(self, test_data):
        """Test removing non-existent treatment."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            treatments_to_remove=['NonExistentTreatment']
        )
        
        assert len(result_df) == len(test_data)
        assert summary['rows_removed_by_metadata'] == 0
    
    def test_status_without_health_report(self, test_data):
        """Test status removal without health report (should skip)."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            statuses_to_remove=['Dead'],
            health_report=None
        )
        
        # Should not crash, status removal should be skipped
        assert len(result_df) == len(test_data)
        assert summary['rows_removed_by_status'] == 0
    
    def test_per_fly_days_missing_exp_day(self, test_data):
        """Test per-fly day removal when Exp_Day column is missing."""
        test_data_no_day = test_data.drop(columns=['Exp_Day'])
        result_df, summary, total_removed = remove_flies(
            test_data_no_day,
            per_fly_remove={'5-ch1': [1, 2]}
        )
        
        # Should not crash, should skip per-fly removal
        assert len(result_df) == len(test_data_no_day)
        assert summary['rows_removed_by_days'] == 0
    
    def test_case_insensitive_matching(self, test_data):
        """Test that matching is case-insensitive."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            genotypes_to_remove=['sss', 'RYE'],  # Mixed case
            sexes_to_remove=['female'],
            treatments_to_remove=['2mm his']
        )
        
        # Should still remove matching flies (case-insensitive)
        assert summary['rows_removed_by_metadata'] > 0
    
    def test_fly_key_case_insensitive(self, test_data):
        """Test that fly keys are matched case-insensitively."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            flies_to_remove=['5-CH1', '5-Ch2']  # Mixed case
        )
        
        # Should remove flies regardless of case
        assert summary['rows_removed_by_fly'] > 0
    
    def test_remove_all_flies(self, test_data):
        """Test removing all flies."""
        # Get all unique fly keys
        all_fly_keys = test_data.apply(
            lambda row: create_fly_key(row['monitor'], row['channel']), axis=1
        ).unique().tolist()
        
        result_df, summary, total_removed = remove_flies(
            test_data,
            flies_to_remove=all_fly_keys
        )
        
        assert len(result_df) == 0
        assert total_removed == len(test_data)
    
    def test_remove_all_by_metadata(self, test_data):
        """Test removing all flies by metadata."""
        result_df, summary, total_removed = remove_flies(
            test_data,
            genotypes_to_remove=['SSS', 'Rye', 'Fmn', 'Iso']  # All genotypes
        )
        
        assert len(result_df) == 0
        assert total_removed == len(test_data)


# ============================================================
#   TESTS: Helper Functions
# ============================================================

class TestHelperFunctions:
    """Test helper functions."""
    
    def test_create_fly_key(self):
        """Test fly key creation."""
        assert create_fly_key(5, 1) == '5-ch1'
        assert create_fly_key(6, 23) == '6-ch23'
        assert create_fly_key(5, '1') == '5-ch1'
        assert create_fly_key(5, 'CH1') == '5-ch1'  # Should handle 'CH' prefix
    
    def test_create_fly_key_case_handling(self):
        """Test fly key creation with various channel formats."""
        assert create_fly_key(5, 'ch1') == '5-ch1'
        assert create_fly_key(5, 'CH1') == '5-ch1'
        assert create_fly_key(5, 1) == '5-ch1'


# ============================================================
#   TESTS: Summary Statistics
# ============================================================

class TestSummaryStatistics:
    """Test that summary statistics are accurate."""
    
    def test_summary_accuracy_single_criterion(self, test_data):
        """Test summary accuracy for single criterion."""
        original_count = len(test_data)
        result_df, summary, total_removed = remove_flies(
            test_data,
            genotypes_to_remove=['SSS']
        )
        
        assert total_removed == original_count - len(result_df)
        assert summary['rows_removed_by_metadata'] == total_removed
    
    def test_summary_accuracy_multiple_criteria(self, test_data, health_report):
        """Test summary accuracy for multiple criteria."""
        original_count = len(test_data)
        result_df, summary, total_removed = remove_flies(
            test_data,
            statuses_to_remove=['Dead'],
            genotypes_to_remove=['SSS'],
            health_report=health_report
        )
        
        assert total_removed == original_count - len(result_df)
        assert (summary['rows_removed_by_status'] + 
                summary['rows_removed_by_metadata']) == total_removed
    
    def test_summary_all_zero_when_no_removals(self, test_data):
        """Test that summary is all zeros when nothing is removed."""
        result_df, summary, total_removed = remove_flies(test_data)
        
        assert summary['flies_removed'] == 0
        assert summary['rows_removed_by_fly'] == 0
        assert summary['rows_removed_by_days'] == 0
        assert summary['rows_removed_by_metadata'] == 0
        assert summary['rows_removed_by_status'] == 0
        assert total_removed == 0


# ============================================================
#   RUN TESTS
# ============================================================

if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])

