import pandas as pd
import sys
from .filters import CheckData, ObjectDictionary, OptimizeResult, sort
from .UtaUtastarFunctions import *
from termcolor import colored
from .UtadisFunctions import *
from scipy.optimize import linprog
from matplotlib import pyplot as plt
from scipy.stats import kendalltau


def UTA(data, alternatives=None, criteria=None, ranking=None, monotonicity=None,
        bestvalues=None, worstvalues=None, intervals=None, method='interior-point',
        epsilon=0.009, delta=0.01, sigma=0):
    filter = CheckData(data)
    if isinstance(data, pd.DataFrame):
        data = filter.ranking_pd()
        data = filter.reposition_ranking_pd()
        data = filter.filter_values_pd()
        ranking = filter.cut_the_ranking()
        data, criteria, alternatives = filter.turn_to_np()
    elif isinstance(data, np.ndarray):
        alternatives = filter.list_filter(alternatives, 0, "alternative")
        criteria = filter.list_filter(criteria, 1, "criterion")
        ranking = filter.ranking_np(ranking)
        ranking, data = filter.filter_values_np(ranking)

    else:
        print(colored(
            "This form of data is not supported by the function, it needs to be either pandas DataFrame or an numpy array",
            "red"))
        sys.exit(-1)

    monotonicity = filter.gradient(monotonicity)
    bestvalues = filter.BestValues(bestvalues, monotonicity)
    worstvalues = filter.WorstValues(worstvalues, monotonicity)
    intervals = filter.Intervals(intervals)

    utilities_multiplier, g = lp_array_1st_part(data, bestvalues, worstvalues, intervals, monotonicity)

    c, A_ineq, A_eq, b_ineq, b_eq = lp_array_2nd_part_UTA(utilities_multiplier, g, ranking, delta, sigma)

    res = linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, method=method)

    A_eq = np.concatenate((A_eq, c), axis=0)
    b_eq = np.append(b_eq, res.fun + epsilon)

    all_c, max_utilities, min_utilities = cost_functionsUTA(data, c, g)

    mean_utilities, max_utilities, min_utilities = calculate_utilitiesUTA(all_c, max_utilities, min_utilities, A_ineq, A_eq, b_ineq, b_eq, method)

    results = np.concatenate((max_utilities, min_utilities), axis=0)

    using_w = 0

    breakpoint = sum([len(criterion_g) - using_w for criterion_g in g])
    Results = np.copy(results[:, 0: breakpoint])

    ASI = ASIcalculation(results, g, using_w)

    alts_utilities, crit_utilities, crit_names, errors = answers1(mean_utilities, g, utilities_multiplier, alternatives)

    results = {
        'g': g,
        'criteria': criteria,
        'crit_names': crit_names,
        'crit_utilities': crit_utilities,
        'all_utilities': Results,
        'alternatives': alternatives,
        'alts_utilities': alts_utilities,
        'errors': errors,
        'ranking': ranking,
        'ASI': ASI}

    return OptimizeResult(results)


def UTASTAR(data, alternatives=None, criteria=None, ranking=None, monotonicity=None, bestvalues=None,
            worstvalues=None, intervals=None, epsilon=0.005, delta=0.05, method='interior-point'):
    filter = CheckData(data)
    if isinstance(data, pd.DataFrame):
        data = filter.ranking_pd()
        data = filter.reposition_ranking_pd()
        data = filter.filter_values_pd()
        ranking = filter.cut_the_ranking()
        data, criteria, alternatives = filter.turn_to_np()
    elif isinstance(data, np.ndarray):
        alternatives = filter.list_filter(alternatives, 0, "alternative")
        criteria = filter.list_filter(criteria, 1, "criterion")
        ranking = filter.ranking_np(ranking)
        ranking, data = filter.filter_values_np(ranking)

    else:
        print(colored(
            "This form of data is not supported by the function, it needs to be either pandas DataFrame or an numpy array",
            "red"))
        sys.exit(-1)

    monotonicity = filter.gradient(monotonicity)
    bestvalues = filter.BestValues(bestvalues, monotonicity)
    worstvalues = filter.WorstValues(worstvalues, monotonicity)
    intervals = filter.Intervals(intervals)

    utilities_multiplier, g = lp_array_1st_part(data, bestvalues, worstvalues, intervals, monotonicity)
    w_array = lp_array_2nd_part_UTASTAR(data, utilities_multiplier, g, ranking)
    errors_array = create_error_array_Utastar(ranking)
    c, A_eq, A_ineq, b_eq, b_ineq = lp_array_3rd_part_UTASTAR(w_array, errors_array, ranking, delta)

    res = linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, method=method)

    A_eq = np.concatenate((A_eq, c), axis=0)
    b_eq = np.append(b_eq, res.fun + epsilon)

    mean_util, results = calculate_utilitiesUTASTAR(g, A_ineq, A_eq, b_eq, b_ineq, method)

    using_w = 1

    breakpoint = sum([len(criterion_g) - using_w for criterion_g in g])
    Results = np.copy(results[:, 0: breakpoint])

    ASI = ASIcalculation(results, g, using_w)

    aggregated_utils = utility_aggregation(mean_util, g)

    alts_utilities, crit_utilities, crit_names, errors = answers1(aggregated_utils, g, utilities_multiplier,
                                                                  alternatives)

    w, w_names = answers2(mean_util, intervals)

    results = {
        'g': g,
        'criteria': criteria,
        'crit_names': crit_names,
        'crit_utilities': crit_utilities,
        'w_names': w_names,
        'w': w,
        'all_w': Results,
        'alternatives': alternatives,
        'alts_utilities': alts_utilities,
        'errors': errors,
        'ranking': ranking,
        'ASI': ASI}

    return OptimizeResult(results)


class UTADIS():
    ''' UTADIS is a multicriteria method used for creating an additive utility function in order to classify
    alternatives into predefined strict groups of preference
    '''

    def __init__(self):
        self.optresults = None
        self.useful = None
        self.predictions = None
        self.estimation = None

    def fit(self, data, groups, pref_order, alternatives=None, criteria=None, monotonicity=None, bestvalues=None,
            worstvalues=None, intervals=None, epsilon=0.005, delta=0.05, sigma=0.1,
            method='interior-point'):
        ''' Use data to create the function that contains the additive utility function, the border
        of the groups and the errors '''
        filter = CheckData(data)
        if isinstance(data, pd.DataFrame):
            data = filter.filter_values_pd()
            data, criteria, alternatives = filter.turn_to_np()
        elif isinstance(data, np.ndarray):
            alternatives = filter.list_filter(alternatives, 0, "alternative")
            criteria = filter.list_filter(criteria, 1, "criterion")
            data = filter.filter_nan_UTADIS()
        else:
            print(colored(
                "This form of data is not supported by the function, it needs to be either pandas DataFrame or an numpy array",
                "red"))
            sys.exit(-1)

        monotonicity = filter.gradient(monotonicity)
        bestvalues = filter.BestValues(bestvalues, monotonicity)
        worstvalues = filter.WorstValues(worstvalues, monotonicity)
        intervals = filter.Intervals(intervals)
        filter.group_consistency(groups, pref_order)

        g = get_g(bestvalues, worstvalues, intervals)

        u_values = get_u_part(data, g, monotonicity)

        A_ineq, b_ineq = linprog_dis_inequalities(u_values, groups, pref_order, delta, sigma)

        A_eq, b_eq = linprog_dis_equalities(A_ineq, g)

        c = linprog_dis_costfunction(A_ineq, u_values, pref_order)

        res = linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, method=method)

        mean_utilities, max_x, min_x = meta_optimization(res, c, A_ineq, b_ineq, A_eq, b_eq, g, method, epsilon)

        results = np.concatenate((max_x, min_x), axis=0)

        using_w = 1

        breakpoint = sum([len(criterion_g) - using_w for criterion_g in g])
        Results = np.copy(results[:, 0: breakpoint])

        ASI = ASIcalculation(results, g, using_w)

        w_values, transitionpoints, errors = x_deconstruction(mean_utilities, g, pref_order)

        crit_utilities = w_aggregation(w_values, g)

        useful = {
            'pref_order': pref_order,
            'monotonicity': monotonicity}

        self.useful = ObjectDictionary(useful)

        results = {
            'g': g,
            'w_values': w_values,
            'all_w': Results,
            'transitionpoints': transitionpoints,
            'errors': errors,
            'crit_utilities': crit_utilities,
            'criteria': criteria,
            'ASI': ASI}
        self.optresults = OptimizeResult(results)

        return self.optresults

    def predict(self, data):
        '''This method predicts the classes for a new set of alternatives with the w values
        of the latest data set.
        data: Numpy array that contains the alternatives with the unknown class.
              The values of every criteria mus be in between the range of best values
              and worst values'''
        filter = CheckData(data)
        if isinstance(data, pd.DataFrame):
            data = filter.filter_values_pd()
            data, criteria, alternatives = filter.turn_to_np()
        elif isinstance(data, np.ndarray):
            data = filter.filter_nan_UTADIS()
        else:
            print(colored(
                "This form of data is not supported by the function, it needs to be either pandas DataFrame or an numpy array",
                "red"))
            sys.exit(-1)

        estimation = estimate_values(self.optresults, self.useful, data)

        self.estimation = estimation

        self.predictions = predict_classes(self.optresults, self.useful, estimation)

        return self.predictions

    def altsestimation(self):
        '''Returns the estimations of every alternative that has been calculated
        on the latest use of the predict method and contains an np array with the
        values'''

        return self.estimation


def criteria_plot(res):
    criterion_index = 0
    for criterion in range(len(res.g)):
        x_index = np.arange(len(res.g[criterion]))

        criterion_title  = res.criteria[criterion]
        criterion_end_of_array = len(res.g[criterion])+criterion_index
        x_values = x_index
        y_values = res.crit_utilities[criterion_index : criterion_end_of_array]
        res.g[criterion]
        plt.plot(x_values,y_values)
        plt.xticks(x_index, res.g[criterion])
        plt.title(criterion_title)
        plt.show()

        criterion_index += len(res.g[criterion])

    return 0


def alternatives_plot(res, x=None):

    if hasattr(res, 'optresults'):
        x_indexes = np.arange(len(res.predictions))
        x_values = res.predictions
        y_values = res.estimation
        labels = res.predictions
        fig, ax = plt.subplots()
        for point in res.optresults.transitionpoints:
            ax.axhline(point, linestyle='--',color='gray', linewidth=1)
    else:
        x_indexes = np.arange(len(res.alternatives))
        x_values = res.alternatives
        y_values = res.alts_utilities
        labels = res.alternatives

    plt.bar(x_indexes, y_values)

    plt.xticks(ticks=x_indexes, labels=labels, rotation='vertical')
    plt.title('Alternatives')
    plt.show()

    return 0

def ktaumetric(res):
    autilities = 1 / res.alts_utilities
    uranking = [sorted(autilities).index(x)+1 for x in autilities]
    tau, p_value = kendalltau(uranking, res.ranking)
    return tau

