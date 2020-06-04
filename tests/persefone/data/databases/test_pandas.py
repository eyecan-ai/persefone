from persefone.data.databases.pandas import PandasDatabase
from persefone.data.databases.pandas import PandasDatabaseGeneratorUnderscoreNotation
from persefone.data.databases.pandas import PandasDatabaseIO
import pandas as pd
import pytest
import numpy as np
from pathlib import Path


class TestPandasDatabase(object):

    @classmethod
    def _generate_item(cls, index):
        """Generate a fake dict item used to emulate databases rows

        :param index: unique identifier (e.g. progressive number)
        :type index: int or str
        :return: useless dictionary
        :rtype: dict
        """
        item = {
            'name': f'item_{index}',
            'a': index * 2 + 4.1454,
            'b': index * 3 + 2.2323,
            'c': index * 15 / 10.2321122112,
            'oddity': 1 if index % 2 == 0 else 0,
            'howmany_5': int(str(index).count('5')),
            'unique_column': PandasDatabase.generate_random_id()
        }
        return item

    @classmethod
    def _generate_items(cls, n):
        """Generates a list of random dict items

        :param n: number of elements
        :type n: int
        :return: list of random elements
        :rtype: int
        """
        items = []
        for i in range(n):
            items.append(cls._generate_item(i))
        return items

    @classmethod
    def _generate_random_database(cls, n=1000, index=None):
        """Generates random PandasDatabase

        :param n: number of elements to generate, defaults to 1000
        :type n: int, optional
        :param index: column index, defaults to None
        :type index: object, optional
        :return: generated PandasDatabase
        :rtype: PandasDatabase
        """
        fake_data_size = n
        items = TestPandasDatabase._generate_items(fake_data_size)
        if len(items) > 0:
            database = PandasDatabase.from_array(items)
        else:
            database = PandasDatabase()
        database.attrs = {
            'attr1': 'custom_database',
            'attr2': 666.66,
            'attr3': [1, 2, 3.3]
        }
        if index is not None and len(items) > 0:
            database.set_index(index)
        return database

    @classmethod
    def _check_different_database(cls, db1, db2):
        assert db1 is not db2, f"Two databases are linked to the same instance!"

    @classmethod
    def _check_if_attrs_are_equal(cls, db1, db2):
        for key, value in db1.attrs.items():
            if key not in db2.attrs:
                return False
        return True
        # TODO: comparison is also in key list!
        # assert db1.attrs == db2.attrs, f"Two database's attrs are different! {db1.attrs} != {db2.attrs}"

    @classmethod
    def _check_if_data_is_dataframe(cls, db: PandasDatabase):
        assert type(db.data) == pd.DataFrame, f"{PandasDatabase.__name__}.data should be a DataFrame!"

    @classmethod
    def _check_if_data_is_not_none(cls, db: PandasDatabase):
        assert db.data is not None, f"{PandasDatabase.__name__}.data should be not none!"

    @classmethod
    def _check_if_data_size_is(cls, db: PandasDatabase, data_size: int):
        current_size = db.size
        assert current_size == data_size, f"{PandasDatabase.__name__}.data size is not {data_size}!"

    @classmethod
    def _verify_assertion_error_with_data(cls, data):
        with pytest.raises(AssertionError):
            database = PandasDatabase(data=data)
            print(database.size)

    def test_pandas_database(self):
        database = PandasDatabase(data=None, attrs=None)
        TestPandasDatabase._check_if_data_is_not_none(database)
        TestPandasDatabase._check_if_data_is_dataframe(database)
        TestPandasDatabase._check_if_data_size_is(database, 0)

        database = PandasDatabase(data=pd.DataFrame())
        TestPandasDatabase._check_if_data_is_not_none(database)
        TestPandasDatabase._check_if_data_is_dataframe(database)
        TestPandasDatabase._check_if_data_size_is(database, 0)

        TestPandasDatabase._verify_assertion_error_with_data(data='fake string')
        TestPandasDatabase._verify_assertion_error_with_data(data=2.22)
        TestPandasDatabase._verify_assertion_error_with_data(data=np.random.uniform(-1, 1, (64, 64, 4)))

    def test_pandas_database_slice_sizes(self):
        """Checks for slices sizes  """
        fake_data_size = 256
        databases = [
            TestPandasDatabase._generate_random_database(fake_data_size, index=None),
            TestPandasDatabase._generate_random_database(fake_data_size, index='name')
        ]

        for database in databases:
            assert len(str(database)) > 0, "String representation of database is empty!"
            TestPandasDatabase._check_if_data_size_is(database, fake_data_size)

            TestPandasDatabase._check_if_data_size_is(database[0], 1)
            TestPandasDatabase._check_if_attrs_are_equal(database, database[0])

            TestPandasDatabase._check_if_data_size_is(database[:10], 10)
            TestPandasDatabase._check_if_attrs_are_equal(database[:10], database)

            TestPandasDatabase._check_if_data_size_is(database[:-10], fake_data_size - 10)
            TestPandasDatabase._check_if_attrs_are_equal(database[:-10], database)

            TestPandasDatabase._check_if_data_size_is(database[-10:], 10)
            TestPandasDatabase._check_if_attrs_are_equal(database[-10:], database)

    def test_pandas_database_slice_single_values(self):
        """Checks for slices single value consistency """
        fake_data_size = 256
        databases = [
            TestPandasDatabase._generate_random_database(fake_data_size, index=None),
            TestPandasDatabase._generate_random_database(fake_data_size, index='name')
        ]

        for database in databases:
            # Check single pick in database
            for picks in [0, 32, 64, 128, -1, fake_data_size, fake_data_size * 2]:
                for column in ['a', 'b']:
                    # Checks if INDEX is valid, if NOT Assertion error must be raised
                    if not database.is_valid_index(picks):
                        with pytest.raises(AssertionError):
                            picked = database[picks: picks + 1]
                        continue
                    else:
                        picked = database[picks:picks + 1]
                        TestPandasDatabase._check_different_database(database, picked)
                        TestPandasDatabase._check_if_attrs_are_equal(database, picked)
                        pick_item = picked.data.iloc[0][column]
                        source_item = TestPandasDatabase._generate_item(picks)[column]
                        assert np.isclose(pick_item, source_item), f"Mismatch between DataFrame values and original "

    def test_pandas_rows_iterator(self):
        """Checks for slices single value consistency """
        fake_data_size = 256
        databases_runs = [
            {'index_type': None, 'db': TestPandasDatabase._generate_random_database(fake_data_size, index=None)},
            {'index_type': 'name', 'db': TestPandasDatabase._generate_random_database(fake_data_size, index='name')}
        ]

        for run in databases_runs:
            database = run['db']
            index_type = run['index_type']
            rows_list = []
            for index, row in database.data.iterrows():
                rows_list.append(row)
                if index_type is None:
                    assert isinstance(index, int), "If not specified, index must be an integer"
                else:
                    assert isinstance(index, str), "If specified, index must be an string"
                assert isinstance(row, pd.Series), "Row is not a series!"
            assert len(rows_list) == database.size, "Size is wrong!"
            assert len(rows_list) == len(database), "Size is wrong!"
            for row in database.data.iterrows():
                assert isinstance(row, tuple), "Simple iteration must returns tuples"

            assert not database.has_duplicated_index(), "Duplicate indices not allowed here!"

    def test_pandas_database_slice_multple_values(self):
        """Checks for slices multiple values consistency """
        fake_data_size = 256
        databases = [
            TestPandasDatabase._generate_random_database(fake_data_size, index=None),
            TestPandasDatabase._generate_random_database(fake_data_size, index='name')
        ]

        for database in databases:
            # Check sliced picks in database
            for r in [(0, 10), (20, 30), (44, 47), (-10, 40), (20, fake_data_size * 20)]:
                column = 'a'
                slice_size = r[1] - r[0]

                # Checks if INDEX is valid, if NOT Assertion error must be raised
                # print("V"*20, r[0], r[1], not database.is_valid_index(r[0]) or not database.is_valid_index(r[1]))
                if not database.is_valid_index(r[0]) or not database.is_valid_index(r[1]):
                    with pytest.raises(AssertionError):
                        slice_items = database[r[0]:r[1]]
                    continue
                else:
                    slice_items = database[r[0]:r[1]]
                    TestPandasDatabase._check_different_database(database, slice_items)
                    TestPandasDatabase._check_if_attrs_are_equal(database, slice_items)
                    sum_target = 0.0
                    sum_source = 0.0
                    for index in range(0, slice_size):
                        sum_target += slice_items.data.iloc[index][column]
                        sum_source += TestPandasDatabase._generate_item(index + r[0])[column]
                    assert np.isclose(sum_target, sum_source), f"Mismatch between DataFrame values and original values on sliced pick."

    def test_pandas_database_split_size(self):
        """Checks for split size consistency """
        fake_data_size = 256
        databases = [
            TestPandasDatabase._generate_random_database(fake_data_size, index=None),
            TestPandasDatabase._generate_random_database(fake_data_size, index='name')
        ]

        for database in databases:
            # Check splits sizes
            for s in [0.2, 0.3, 0.5, 0.8, 0.75411, 0, -1, 1.0, 1000.]:
                d0, d1 = database.split(s)
                TestPandasDatabase._check_different_database(database, d0)
                TestPandasDatabase._check_different_database(database, d1)
                TestPandasDatabase._check_if_attrs_are_equal(database, d0)
                TestPandasDatabase._check_if_attrs_are_equal(database, d1)
                s0 = int(fake_data_size * np.clip(s, 0, 1))
                s1 = database.size - s0
                assert d0.size == s0, f"{d0.size} != {s0}, database size after split is wrong!"
                assert d1.size == s1, f"{d1.size} != {s1}, database size after split is wrong!"

    def test_pandas_database_multiple_split_size(self):
        """Checks for split size consistency """
        fake_data_size = 256
        databases = [
            TestPandasDatabase._generate_random_database(fake_data_size, index=None),
            TestPandasDatabase._generate_random_database(fake_data_size, index='name')
        ]

        divisions = [
            [0.9],
            [0.8, 0.2],
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1, 0.01],
        ]

        for database in databases:
            # Check splits sizes
            for percentages in divisions:
                percentages_sum = np.array(percentages).sum()
                if percentages_sum <= 1.0:
                    chunks = database.splits(percentages)
                    assert len(chunks) == len(percentages), "Chunks number is different from divisions number"

                    for a in range(len(chunks)):
                        for b in range(len(chunks)):
                            if a != b:
                                assert not chunks[a].check_intersection(chunks[b]), "Chunks must be orthogonals!"
                else:
                    with pytest.raises(AssertionError):
                        chunks = database.splits(percentages)

    def test_pandas_database_multiple_split_as_dict(self):
        """Checks for split size consistency """
        fake_data_size = 256
        databases = [
            TestPandasDatabase._generate_random_database(fake_data_size, index=None),
            TestPandasDatabase._generate_random_database(fake_data_size, index='name')
        ]

        divisions_map = [
            {'train': 0.9},
            {'train': 0.8, 'test': 0.1, 'val': 0.1},
            {'one': 0.1, 'two': 0.1, 'three': 0.1, 'remains': 0.7},
        ]

        for check_integrity in [True, False]:
            for database in databases:
                # Check splits sizes
                for percentages_map in divisions_map:
                    percentages_sum = np.array(list(percentages_map.values())).sum()
                    if percentages_sum <= 1.0:
                        chunks = database.splits_as_dict(percentages_map, integrity=check_integrity)
                        assert len(chunks) == len(percentages_map), "Chunks number is different from divisions number"

                        for a, _ in chunks.items():
                            for b, _ in chunks.items():
                                if a != b:
                                    assert not chunks[a].check_intersection(chunks[b]), "Chunks must be orthogonals!"

                        if check_integrity:
                            sumup = 0
                            for _, chunk in chunks.items():
                                sumup += chunk.size
                            assert sumup == database.size, "Size consistency fails!"
                    else:
                        with pytest.raises(AssertionError):
                            chunks = database.splits_as_dict(percentages_map)

    def test_pandas_database_split_indices(self):
        """Checks for split indices consistency """
        fake_data_size = 256
        databases = [
            TestPandasDatabase._generate_random_database(fake_data_size, index=None),
            TestPandasDatabase._generate_random_database(fake_data_size, index='name')
        ]

        for database in databases:
            # Verify reset index after split. RESET is set to FALSE
            reset = False
            for s in [0.2, 0.3, 0.5, 0.8, 0.75411, 0, -1, 1.0, 1000.]:
                d0, d1 = database.split(s)

                TestPandasDatabase._check_different_database(database, d0)
                TestPandasDatabase._check_different_database(database, d1)
                TestPandasDatabase._check_if_attrs_are_equal(database, d0)
                TestPandasDatabase._check_if_attrs_are_equal(database, d1)
                d0_indices = d0.data.index.to_numpy()
                d1_indices = d1.data.index.to_numpy()

                sum_indices = np.concatenate([d0_indices, d1_indices])
                d_indices = database.data.index.to_numpy()
                if not reset or s <= 0 or s >= 1.0:
                    assert np.array_equal(sum_indices, d_indices), f"database indices wrong after split with Reset={reset}"
                else:
                    assert not np.array_equal(sum_indices, d_indices), f"database indices wrong after split with Reset={reset}"

    def test_shuffle(self):
        """ Checks if shuffle operation is consistent """
        fake_data_size = 10000
        databases = [
            TestPandasDatabase._generate_random_database(fake_data_size, index=None),
            TestPandasDatabase._generate_random_database(fake_data_size, index='name')
        ]

        for database in databases:
            s_database = database.shuffle(seed=666)
            TestPandasDatabase._check_different_database(database, s_database)
            TestPandasDatabase._check_if_attrs_are_equal(database, s_database)
            assert not database.data.index.equals(s_database.data.index), "After shuffling indices must be different"

    def test_rebuild_indices(self):
        """ Checks rebuild indices """
        fake_data_size = 10000
        databases = [
            TestPandasDatabase._generate_random_database(fake_data_size, index=None),
            TestPandasDatabase._generate_random_database(fake_data_size, index='name')
        ]

        for database in databases:
            database_re = database.rebuild_indices()
            assert self._check_if_attrs_are_equal(database_re, database), "After rebuild indices attrs was lost!"

    def test_summation(self):
        """ Checks if sums among PandasDatabase are consistent """
        fake_data_size = 10000

        indices = [None, 'name']

        for index in indices:
            database1 = TestPandasDatabase._generate_random_database(fake_data_size, index=index).rebuild_indices()
            database2 = TestPandasDatabase._generate_random_database(fake_data_size, index=index).rebuild_indices()
            database3 = TestPandasDatabase._generate_random_database(fake_data_size, index=index).rebuild_indices()
            empty_dataset = TestPandasDatabase._generate_random_database(0, index=index)

            # SUM D1 + D1
            database_sum = database1 + database1
            TestPandasDatabase._check_if_data_size_is(database_sum, fake_data_size * 2)
            TestPandasDatabase._check_different_database(database1, database_sum)
            assert database_sum.has_duplicated_index(), "Summing a database with itself needs to conaint duplicate indices!"

            # SUM D1 + D2
            database_sum = database1 + database2
            TestPandasDatabase._check_if_data_size_is(database_sum, fake_data_size * 2)
            TestPandasDatabase._check_different_database(database1, database_sum)
            TestPandasDatabase._check_different_database(database2, database_sum)
            assert not database_sum.has_duplicated_index(), "Summing two reindexed databases needs to avoid duplicate indices!"

            # SUM D1 + D2 + D3
            database_sum = database1 + database2 + database3
            TestPandasDatabase._check_if_data_size_is(database_sum, fake_data_size * 3)
            TestPandasDatabase._check_different_database(database1, database_sum)
            TestPandasDatabase._check_different_database(database2, database_sum)
            TestPandasDatabase._check_different_database(database3, database_sum)
            assert not database_sum.has_duplicated_index(), "Summing two reindexed databases needs to avoid duplicate indices!"

            # Empty datasets
            fake = empty_dataset + database1
            assert fake.size == fake_data_size, "Summing with empty dataset as first addendum is wrong!"
            fake = database1 + empty_dataset
            assert fake.size == fake_data_size, "Summing with empty dataset as second addendum is wrong!"
            database1 += empty_dataset
            assert database1.size == fake_data_size, "Summing with empty dataset as second addendum is wrong!"

    def test_intersection(self):
        """ Test intersection among datasets """
        fake_data_size = 10000

        indices = [None, 'name']

        for index in indices:
            database1 = TestPandasDatabase._generate_random_database(fake_data_size, index=index).rebuild_indices()
            database2 = TestPandasDatabase._generate_random_database(fake_data_size, index=index).rebuild_indices()

            c = 'a'
            assert database1.check_intersection(database2, subset=[c]), f"Intersection on column {c} must not be empty!"
            assert database1.check_intersection(database1, subset=[c]), f"Intersection on non unique column {c} must not be empty!"

            c = 'unique_column'
            assert not database1.check_intersection(database2, subset=[c]), f"Intersection on column {c} must be empty!"
            assert database1.check_intersection(database1, subset=[c]), f"Intersection on non unique column {c} must not be empty!"

            assert not database1.check_intersection(database2, subset='index'), "Intersection with rebuilt indices has to be void!"
            assert database1.check_intersection(database1, subset='index'), "Intersection with itself has not to be empty!"
            assert database1.check_intersection(database1[:10], subset='index'), "Intersection with a subset of it has not to be empty!"
            assert database1.check_intersection(database1[5], subset='index'), "Intersection with a subset of it has not to be empty!"
            assert database1.check_intersection(database1[-5:], subset='index'), "Intersection with a subset of it has not to be empty!"

    def test_grouping(self):
        """ Tests grouping counter feature """
        fake_data_size = 10015  # must to be odd!!
        indices = [None, 'name']

        for index in indices:
            database1 = TestPandasDatabase._generate_random_database(fake_data_size, index=index).rebuild_indices()

            count_map = database1.count_grouped_values('oddity')
            assert count_map['oddity'][0] != count_map['oddity'][1], "Oddity Test column grouped values must to be different"

            count_map = database1.count_grouped_values('howmany_5')
            assert count_map['howmany_5'][2] == 486, "HOWMANY_5 Test column grouped value has to count 486 for items with 2 '5' within"

    def test_unique(self):
        """ Tests grouping counter feature """
        fake_data_size = 1000
        indices = [None, 'name']

        for index in indices:
            database1 = TestPandasDatabase._generate_random_database(fake_data_size, index=index).rebuild_indices()

            assert len(database1.unique('index')) == database1.size, "Unique indices size must be equale to indices size itself"

            c = 'oddity'
            assert len(database1.unique(c)) != database1.size, f"Unique column {c} size must be lesser than total items"

            c = 'howmany_5'
            assert len(database1.unique(c)) != database1.size, f"Unique column {c} size must be lesser than total items"

    def test_filter_rows_by_column_values(self):
        """ Test for filtering by column function """
        fake_data_size = 10000  # must be 10000! don't change
        indices = [None, 'name']

        for index in indices:
            database1 = TestPandasDatabase._generate_random_database(fake_data_size, index=index).rebuild_indices()

            # Checks for oddity consitency by filtering rows
            half = int(fake_data_size / 2)
            assert database1.filter_rows_by_column_values('oddity', [1]).size == half, f"odds items count must be {half}"
            assert database1.filter_rows_by_column_values('oddity', 1).size == half, f"odds items count must be {half}"
            assert database1.filter_rows_by_column_values('oddity', [0]).size == half, f"evens items count must be {half}"
            assert database1.filter_rows_by_column_values('oddity', 0).size == half, f"evens items count must be {half}"
            TestPandasDatabase._check_different_database(database1, database1.filter_rows_by_column_values('oddity', 0))
            TestPandasDatabase._check_different_database(database1, database1.filter_rows_by_column_values('oddity', [0]))
            assert database1.filter_rows_by_column_values('oddity', [0, 1]).size == fake_data_size, f"whole items count is wrong"

            # Counts items filtered by test column "HOWMANY_5". counting rows it 0 or 1 '5' present, or the opposite
            h5count = 6561 + 2916
            assert database1.filter_rows_by_column_values(
                'howmany_5', [0, 1]).size == h5count, f"items with 0 or 1 '5' count is wrong!"
            assert database1.filter_rows_by_column_values(
                'howmany_5', [0, 1], negate=True).size == fake_data_size - h5count, f"items with more than 0 or 1 '5' count is wrong!"
            assert database1.filter_rows_by_column_values(
                'howmany_5', [0, 1, 'aa', 11, -23]).size == h5count, f"items with 0 or 1 '5' count is wrong!"

            # Empty
            assert database1.filter_rows_by_column_values('howmany_5', [11]).size == 0, 'Database must be empty'
            assert database1.filter_rows_by_column_values('howmany_5', [22]).size == 0, 'Database must be empty'

    def test_split_by_column_values(self):

        fake_data_size = 10005  # must be 10005! don't change
        odds_count = int(fake_data_size / 2) + 1
        evens_count = int(fake_data_size / 2)
        indices = [None, 'name']

        for index in indices:
            database1 = TestPandasDatabase._generate_random_database(fake_data_size, index=index).rebuild_indices()

            # Checks for oddity consitency by filtering rows
            odds, evens = database1.split_by_column_values('oddity', [1, 0])
            assert odds.size == odds_count, "odds size wrong!"
            assert evens.size == evens_count, "evens size wrong!"
            TestPandasDatabase._check_different_database(odds, evens)
            TestPandasDatabase._check_different_database(odds, database1)
            TestPandasDatabase._check_different_database(evens, database1)
            assert not odds.check_intersection(evens, subset='oddity'), "intersection of oddity is not empty!"
            assert not evens.check_intersection(odds, subset='oddity'), "intersection of oddity is not empty!"

            odds, evens = database1.split_by_column_values('oddity', [[1], 0])
            assert odds.size == odds_count, "odds size wrong!"
            assert evens.size == evens_count, "evens size wrong!"

            odds, evens = database1.split_by_column_values('oddity', [[1], [0]])
            assert odds.size == odds_count, "odds size wrong!"
            assert evens.size == evens_count, "evens size wrong!"

            # fake multiple values added other than 0,1
            odds, evens = database1.split_by_column_values('oddity', [[1, 'a', 'f'], [0, 4, 5]])
            assert odds.size == odds_count, "odds size wrong!"
            assert evens.size == evens_count, "evens size wrong!"

            # Splits items based on howmany_5 columns (counting how many '5' chars in name)
            g0, g1, g2, g3, g4 = database1.split_by_column_values('howmany_5', [0, 1, 2, 3, 4])

            assert g0.size + g1.size + g2.size + g3.size + g4.size == database1.size, "Parts size is different from total size"

            assert not g0.check_intersection(g1, subset='howmany_5'), 'Intersection must be empty!'
            assert g0.check_intersection(g0, subset='howmany_5'), 'Intersection must be not empty!'
            assert not g1.check_intersection(g2, subset='howmany_5'), 'Intersection must be empty!'
            assert not g2.check_intersection(g3, subset='howmany_5'), 'Intersection must be empty!'
            assert not g3.check_intersection(g4, subset='howmany_5'), 'Intersection must be empty!'
            assert not g4.check_intersection(g0, subset='howmany_5'), 'Intersection must be empty!'
            assert database1.size == fake_data_size, "Original database is not preserved!"

            # Splits impossibile values
            g11, g22 = database1.split_by_column_values('howmany_5', [11, 22])
            assert g11.size == 0, "Database must be empty!"
            assert g22.size == 0, "Database must be empty!"

            # raise sumup exception
            with pytest.raises(AssertionError):
                g0, g1, g2, g3 = database1.split_by_column_values('howmany_5', [0, 1, 2, 3], check_sizes=True)

    def test_query(self):

        fake_data_size = 10005  # must be 10005! don't change
        odds_count = int(fake_data_size / 2) + 1
        evens_count = int(fake_data_size / 2)
        indices = ['name']

        for index in indices:
            database1 = TestPandasDatabase._generate_random_database(fake_data_size, index=index).rebuild_indices()

            print(database1.size)

            assert isinstance(database1.query("oddity == 1"), PandasDatabase), "What is it?"
            assert isinstance(database1.query("oddity == 0"), PandasDatabase), "What is it?"
            assert database1.query("oddity == 1").size == odds_count, "Odds count is wrong"
            assert database1.query("oddity in [1]").size == odds_count, "Odds count is wrong"
            assert database1.query(f"oddity in {[1]}").size == odds_count, "Odds count is wrong"
            assert database1.query("oddity == 0").size == evens_count, "Evens count is wrong"
            assert database1.query("oddity in [0]").size == evens_count, "Evens count is wrong"
            assert database1.query(f"oddity in {[0]}").size == evens_count, "Evens count is wrong"

            assert database1.query("oddity == 1").query("oddity == 0").size == 0, "Database must be empty!"
            assert database1.query("oddity == 0").query("oddity == 1").size == 0, "Database must be empty!"
            assert database1.query("oddity == 1").query("oddity == 1").size == odds_count, "Database must be not empty!"

            assert TestPandasDatabase._check_if_attrs_are_equal(database1, database1.query("oddity == 1")), "Attributes lost after query!"
            assert TestPandasDatabase._check_if_attrs_are_equal(database1, database1.query("oddity == 0")), "Attributes lost after query!"

            assert database1.query("oddity > 1").size == 0, "Oddity cannot be > 1"

            queries = [
                'oddity == 1',
                'oddity in [1]',
                'a > -0.1',
                'b > -0.1',
                'c > -0.1'
            ]
            assert isinstance(database1.query_list(queries), PandasDatabase), "What is it?"
            assert database1.query_list(queries).size == odds_count, "Size is wrong!"


class TestPandasDatabaseIO(object):

    @pytest.fixture(scope="session")
    def dataset_file(self, tmpdir_factory):
        fn = tmpdir_factory.mktemp("data").join("_database_file.data")
        return fn

    @pytest.fixture(scope="session")
    def dataset_file_csv(self, tmpdir_factory):
        fn = tmpdir_factory.mktemp("data").join("_database_file.csv")
        return fn

    def test_pickle_io(self, dataset_file, minimnist_folder):

        columns_lambdas = {
            'label': lambda x: np.loadtxt(x).astype(int).tolist(),
            'points': lambda x: np.loadtxt(x).tolist(),
        }

        database = PandasDatabaseGeneratorUnderscoreNotation.generate_from_folder(folder=minimnist_folder, columns_lambdas=columns_lambdas)
        database.attrs['base_folder'] = 'minimnist'

        PandasDatabaseIO.save_pickle(database, dataset_file)

        metadata_file = PandasDatabaseIO.build_metadata_filename_from_original_filename(dataset_file)

        assert Path(dataset_file).exists(), "Metadatafile does not exist!"
        assert Path(metadata_file).exists(), "Metadatafile does not exist!"

        database_reloaded = PandasDatabaseIO.load_pickle(dataset_file)

        assert database.data.equals(database_reloaded.data), "Datasets differ after reloading!"
        assert TestPandasDatabase._check_if_attrs_are_equal(database, database_reloaded), "Datasets metadatas differ after reloading!"

        with pytest.raises(OSError):
            PandasDatabaseIO.load_pickle(str(dataset_file) + "_impossiblesuffix_")

        with pytest.raises(OSError):
            database = PandasDatabaseGeneratorUnderscoreNotation.generate_from_folder(
                folder=str(minimnist_folder) + "IMPOSSIBLE_TAG#",
                columns_lambdas=columns_lambdas
            )

    def test_csv_io(self, dataset_file_csv, minimnist_folder):

        # parent_folder = minimnist_folder.parent

        columns_lambdas = {
            'label': lambda x: np.loadtxt(x).astype(int).tolist(),
            'points': lambda x: np.loadtxt(x).tolist(),
        }

        database = PandasDatabaseGeneratorUnderscoreNotation.generate_from_folder(folder=minimnist_folder, columns_lambdas=columns_lambdas)
        database.attrs['base_folder'] = 'minimnist'

        PandasDatabaseIO.save_csv(database, dataset_file_csv)

        metadata_file = PandasDatabaseIO.build_metadata_filename_from_original_filename(dataset_file_csv)

        assert Path(dataset_file_csv).exists(), "Metadatafile does not exist!"
        assert Path(metadata_file).exists(), "Metadatafile does not exist!"

        database_reloaded = PandasDatabaseIO.load_csv(dataset_file_csv)
        assert TestPandasDatabase._check_if_attrs_are_equal(database, database_reloaded), "Datasets metadatas differ after reloading!"

        void_metadata = PandasDatabaseIO.load_metadata(dataset_file_csv + "_NPASDASDASODMASPDASDAS")
        assert void_metadata == {}, "Strange metadata loaded with an invalid filename!"

    def test_csv_rename(self, dataset_file, minimnist_folder):

        columns_lambdas = {
            'label': lambda x: np.loadtxt(x).astype(int).tolist(),
            'points': lambda x: np.loadtxt(x).tolist(),
        }

        database = PandasDatabaseGeneratorUnderscoreNotation.generate_from_folder(folder=minimnist_folder, columns_lambdas=columns_lambdas)
        PandasDatabaseIO.save_csv(database, dataset_file)

        old_index = 'id'
        new_index = 'new_index_name'

        PandasDatabaseIO.rename_index_inplace(dataset_file, old_index=old_index, new_index=new_index)
        # PandasDatabaseIO.rename_index_inplace(dataset_file, old_index='label', new_index='new_label')

        database_r = PandasDatabaseIO.load_csv(dataset_file)

        assert new_index in database_r.data.index.names, "New id is wrong!"
        assert old_index not in database_r.data.index.names, "New id is wrong!"

        PandasDatabaseIO.rename_index_inplace(dataset_file, old_index='impossible_column_NAM3', new_index='newname')
