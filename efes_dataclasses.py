import numpy as np
import pandas as pd
from dataclasses import dataclass, fields
from typing import Union, Optional, List

""" DE-/SERIALIZATION """
def to_json_dict(cls, obj, add_type_info=False, exclude=[]):
    json_dict = {}

    if add_type_info:
        json_dict['type_info'] = cls.__name__

    for field in fields(cls):
        name = field.name
        if name in exclude:
            continue
        value = getattr(obj, name)
        if value is None:
            continue
        try:
            value = value.to_json_dict()
        except AttributeError:
            pass

        if isinstance(value, pd.Series):
            value = value.tolist()

        if isinstance(value, list):
            try:
                value = [v.to_json_dict() for v in value]
            except AttributeError:
                pass

        if isinstance(value, np.ndarray):
            value = str(value.tolist())

        json_dict[name] = value
    return json_dict


def write_to_json(obj, filename, indent=2, exclude=[]):
    import json

    if isinstance(obj, list):
        json_obj = [o.to_json_dict(add_type_info=True, exclude=exclude) for o in obj]
    elif isinstance(obj, dict):
        json_obj = {key: o.to_json_dict(add_type_info=True, exclude=exclude) for (key, o) in obj.items()}
    elif isinstance(obj, pd.DataFrame):
        json_obj = obj.to_json()
    else:
        json_obj = obj.to_json_dict(add_type_info=True, exclude=exclude)

    json_str = json.dumps(json_obj, indent=indent)\
        .replace('], [', f'],\n{" " * (indent + 5)}[')\
        .replace('"[', '[')\
        .replace(']"', ']')\
        .replace('True', 'true')\
        .replace('False', 'false')\
        .replace('Infinity', '"inf"')\
        .replace('nan,', '"nan",')

    with open(f"{filename}.json", "w+") as file:
        file.write(json_str)

def from_json_dict(json_obj):
    registry = {
        'Results': Results,
        'results': Results,
        'AnalysisResults': AnalysisResults,
        'analysis_results': AnalysisResults,
        'QueryResults': QueryResults,
        'DataInput': DataInput,
        'data_input': DataInput,
        'QueryInput': QueryInput,
        'query_input': QueryInput,
        'PhaseData': PhaseData,
        'ParameterStudyResults': ParameterStudyResults
    }

    keys = None
    objs = None

    if isinstance(json_obj, list):
        keys = np.arange(0, len(json_obj))
        objs = json_obj
    elif isinstance(json_obj, dict):
        keys = json_obj.keys()
        objs = json_obj.values()

    if 'type_info' in keys:
        return registry[json_obj['type_info']](**json_obj)

    for key, obj in zip(keys, objs):
        if 'type_info' in obj:
            print(f"'Constructing {obj['type_info']}")
            json_obj[key] = registry[obj['type_info']](**obj)
        else:
            for name, value in obj.items():
                if name in registry:
                    print(f'{name} found in registry')
                    obj[name] = registry[name](**value)
                    break
                else:
                    print(f'{name} not found in registry')

    return json_obj

def read_from_json(filename):
    import json
    with open(f"{filename}.json", "r+") as file:
        json_obj = json.load(file)

    return from_json_dict(json_obj)


def pickle(obj, filename):
    import pickle
    if filename[-7:] != '.pickle':
        filename = filename + '.pickle'
    pickle.dump(obj, open(filename, 'wb'))
    #print(f'Object pickled to {filename}')
def unpickle(filename):
    import pickle
    if filename[-7:] != '.pickle':
        filename = filename + '.pickle'
    obj = pickle.load(open(filename, 'rb'))
    #print(f'Object unpickled from {filename}')
    return obj


""" DATA CLASSES """
@dataclass
class DataInput:
    """Class to store the input data"""
    power_generation: Union[np.ndarray, float, int] = None
    power_demand: Union[np.ndarray, float, int] = None

    delta_time_step: float = None

    power_used_generation: np.ndarray = None
    power_covered_demand: np.ndarray = None
    power_residual_generation: np.ndarray = None

    power_max_discharging: float = np.inf
    power_max_charging: float = np.inf
    efficiency_direct_usage: float = 1.0
    efficiency_discharging: float = 1.0
    efficiency_charging: float = 1.0

    def set_attr(self, name, value):
        if isinstance(value, list):
            value = np.array(value)

        try:
            value = value.to_numpy()
        except AttributeError:
            pass

        self.__setattr__(name, value)

    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        [self.set_attr(name, value) for (name, value) in kwargs.items()]

    def to_json_dict(self, add_type_info=False, exclude=[]):
        return to_json_dict(DataInput, self, add_type_info, exclude)


@dataclass
class QueryInput:
    """Class for query inputs"""
    self_sufficiency_target: Optional[Union[float, np.ndarray]] = None
    self_consumption_target: Optional[Union[float, np.ndarray]] = None
    energy_additional_target: Optional[Union[float, int, np.ndarray]] = None
    capacity_target: Optional[Union[float, int, np.ndarray]] = None

    def set_attr(self, name, value):
        if isinstance(name, list):
            value = np.array(value)
        if isinstance(name, pd.Series):
            value = value.to_numpy()
        self.__setattr__(name, value)

    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        [self.set_attr(name, value) for (name, value) in kwargs.items()]

    def to_json_dict(self, add_type_info=False, exclude=[]):
        return to_json_dict(QueryInput, self, add_type_info, exclude)


class Phase:
    """A class to describe a balancing phase consisting of  energy packets for excess and deficit"""
    def __init__(self, energy_excess: float, energy_deficit: float, id: int = None):
        self.id = id

        self.starts_excess = np.array([0.])
        self.starts_deficit = np.array([0.])
        self.energy_excess = np.array([energy_excess])
        self.energy_deficit = np.array([energy_deficit])
        self.excess_balanced = np.array([False])
        self.deficit_balanced = np.array([False])

        self.excess_ids = np.array([self.id])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = f'Phase {self.id}:\n'
        s += f'starts_excess={self.starts_excess.__str__()}, energy_excess={self.energy_excess.__str__()}, excess_balanced={self.excess_balanced.__str__()}, excess_ids={self.excess_ids.__str__()}\n'
        s += f'starts_deficit={self.starts_deficit.__str__()}, energy_deficit={self.energy_deficit.__str__()}, deficit_balanced={self.deficit_balanced.__str__()}\n'
        return s

    def __eq__(self, other):
        if not isinstance(other, Phase):
            return NotImplemented
        return (np.array_equal(self.starts_excess, other.starts_excess) and
                np.array_equal(self.starts_deficit, other.starts_deficit) and
                np.array_equal(self.energy_excess, other.energy_excess) and
                np.array_equal(self.energy_deficit, other.energy_deficit) and
                np.array_equal(self.excess_balanced, other.excess_balanced) and
                np.array_equal(self.deficit_balanced, other.deficit_balanced))


@dataclass
class PhaseData:
    """Class for phase specific data"""
    power: Optional[np.ndarray] = None
    duration: Optional[np.ndarray] = None
    energy: Optional[np.ndarray] = None

    def set_attr(self, name, value):
        if isinstance(name, list):
            value = np.array(value)
        if isinstance(name, pd.Series):
            value = value.to_numpy()
        self.__setattr__(name, value)

    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        [self.set_attr(name, value) for (name, value) in kwargs.items()]

    def to_json_dict(self, add_type_info=False, exclude=[]):
        return to_json_dict(PhaseData, self, add_type_info, exclude)


@dataclass
class DimensioningResults:
    """Class for the dimensioning data"""
    capacity: Union[float, np.ndarray] = None
    energy_additional: Union[float, np.ndarray] = None
    self_sufficiency: Union[float, np.ndarray] = None
    self_consumption: Union[float, np.ndarray] = None

    gain: Union[float, np.ndarray] = None

    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        [self.set_attr(name, value) for (name, value) in kwargs.items()]

    def to_json_dict(self, add_type_info=False, exclude=[]):
        return to_json_dict(DimensioningResults, self, add_type_info, exclude)


@dataclass
class AnalysisResults(DimensioningResults):
    """Class for the results of the analysis (without the query specific results)."""
    data_input: DataInput = None

    energy_used_generation: float = None
    energy_covered_demand: float = None
    energy_demand: float = None
    energy_generation: float = None

    power_residual_generation_clipped: np.ndarray = None
    self_sufficiency_initial: float = None
    self_consumption_initial: float = None

    time_total: float = None

    starts_phases: np.ndarray = None
    lengths_phases: np.ndarray = None
    values_phases: np.ndarray = None
    N_phases: int = None

    energy_excess_wo_efficiency: np.ndarray = None
    energy_excess: np.ndarray = None

    energy_deficit_wo_efficiency: np.ndarray = None
    energy_deficit_wo_debt: np.ndarray = None
    energy_deficit: np.ndarray = None

    debt_comment: str = None
    debt: np.ndarray = None
    overflow: float = None

    effectiveness_local: np.ndarray = None

    phases: List[Phase] = None

    phase_data_deficit: List[PhaseData] = None
    phase_data_excess: List[PhaseData] = None

    capacity_max: float = None
    energy_additional_max: float = None
    self_sufficiency_max: float = None
    self_consumption_max: float = None

    def __init__(self, **kwargs):
        self.update(**kwargs)

    def set_attr(self, name, value):
        if name == 'data_input' and not isinstance(value, DataInput):
            value = DataInput(**value)

        if name == 'phase_data_deficit' or name == 'phase_data_excess':
            for i, d in enumerate(value):
                if not isinstance(d, PhaseData):
                    d = PhaseData(**d)
                value[i] = d
        else:
            if isinstance(value, list):
                value = np.array(value)
            if isinstance(name, pd.Series):
                value = value.to_numpy()

        #print(f'Setting {name} with {value}')
        self.__setattr__(name, value)

    def to_json_dict(self, add_type_info=False, exclude=[]):
        return to_json_dict(AnalysisResults, self, add_type_info, exclude)


@dataclass
class QueryResults(DimensioningResults):
    """Class for additional query results"""
    query_input: QueryInput = None
    effectiveness_local: np.ndarray = None
    effectiveness: np.ndarray = None
    gain: np.ndarray = None
    gain_per_day: np.ndarray = None

    def __init__(self, **kwargs):
        self.update(**kwargs)

    def set_attr(self, name, value):
        if name == 'query_input' and not isinstance(value, QueryInput):
            value = QueryInput(**value)

        self.__setattr__(name, value)

    def to_json_dict(self, add_type_info=False, exclude=[]):
        return to_json_dict(QueryResults, self, add_type_info, exclude)


@dataclass
class Results:
    """Class for storing collected results"""
    analysis_results: AnalysisResults = None
    query_results: List[QueryResults] = None

    def set_attr(self, name, value):
        if name == 'analysis_results' and not isinstance(value, AnalysisResults):
            value = AnalysisResults(**value)
        if name == 'query_results' and not isinstance(value[0], QueryResults):
            value = [QueryResults(**v) for v in value]

        self.__setattr__(name, value)

    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        [self.set_attr(name, value) for (name, value) in kwargs.items()]

    def to_json_dict(self, add_type_info=False, exclude=[]):
        if self.query_results is None or 'query_results' in exclude:
            return to_json_dict(Results, self, add_type_info=add_type_info, exclude=exclude)

        tmp_query_results = self.query_results
        self.query_results = None
        tmp_json_dict = to_json_dict(Results, self, add_type_info=add_type_info, exclude=exclude)
        tmp_json_dict['query_results'] = [to_json_dict(QueryResults, query_result, add_type_info=False, exclude=exclude) for query_result in tmp_query_results]
        self.query_results = tmp_query_results
        return tmp_json_dict

@dataclass
class ParameterStudyResults:
    parameter_variation: pd.DataFrame = None
    results: List[Results] = None

    def __init__(self, **kwargs):
        self.update(**kwargs)

    def set_attr(self, name, value):
        if name == 'parameter_variation' and isinstance(value, dict):
            value = pd.DataFrame(data=value)
        self.__setattr__(name, value)

    def update(self, **kwargs):
        [self.set_attr(name, value) for (name, value) in kwargs.items()]

    def to_json_dict(self, add_type_info=False, exclude=[]):
        tmp_parameter_variation = self.parameter_variation
        self.parameter_variation = None
        tmp_json_dict = to_json_dict(ParameterStudyResults, self, add_type_info=add_type_info, exclude=exclude)
        tmp_json_dict['parameter_variation'] = tmp_parameter_variation.to_dict(orient='list')
        self.parameter_variation = tmp_parameter_variation
        return tmp_json_dict