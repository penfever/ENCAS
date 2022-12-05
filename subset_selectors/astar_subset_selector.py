import numpy as np
from pymoo.algorithms.nsga3 import ReferenceDirectionSurvival, get_extreme_points_c, get_nadir_point, \
    associate_to_niches, calc_niche_count, niching
from pymoo.factory import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.spatial import Delaunay, ConvexHull
from subset_selectors.base_subset_selector import BaseSubsetSelector
from matplotlib import pyplot as plt

class MHASubsetSelector(BaseSubsetSelector):
    '''
    Multi-heuristic A* algorithm
    '''
    def __init__(self, n_select, **kwargs):
        super().__init__()
        self.n_select = n_select
        ref_dirs = get_reference_directions("riesz", kwargs['n_objs'], 100)
        self.selector = ReferenceDirectionSurvivalMy(ref_dirs)


    def select(self, archive, objs):
        # objs_cur shape is (n_archs, n_objs)
        objs = np.copy(objs)

        n_total = objs.shape[0]
        if n_total > self.n_select:
            indices_selected = self.selector._do(None, objs, self.n_select)
        else:
            indices_selected = [True] * n_total
        print(f'Selection complete: Selected {np.sum(indices_selected)} indices from {n_total} possibilities')
        return indices_selected


class ReferenceDirectionSurvivalMy(ReferenceDirectionSurvival):
    '''
    Modified to work with np archives instead of the "population" structures.
    Behaviourally is the same as ReferenceDirectionSurvival.
    '''
    def __init__(self, ref_dirs):
        super().__init__(ref_dirs)
        self.ref_dirs = ref_dirs
        self.extreme_points = None
        self.intercepts = None
        self.nadir_point = None
        self.opt = None
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)
        self.worst_point = np.full(ref_dirs.shape[1], -np.inf)

    def _do(self, problem, objs, n_survive, D=None, **kwargs):
        #Number of archs to examine
        n_total = objs.shape[0]
        assert n_survive > len(objs[0]), "This algorithm requires that the number of archs to survive be at least the number of objectives"
        #initialize all selections to false
        indices_selected = np.array([False] * n_total)

        # find or usually update the new ideal point - from feasible solutions
        self.ideal_point = np.min(np.vstack((self.ideal_point, objs)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, objs)), axis=0)

        # calculate the fronts of the population
        fronts, rank = NonDominatedSorting().do(objs, return_rank=True, n_stop_if_ranked=n_survive)
        non_dominated, last_front = fronts[0], fronts[-1]
        print(f"non dominated is {non_dominated}, number of fronts is {len(fronts)}")

        # find the extreme points for normalization
        self.extreme_points = get_extreme_points_c(objs[non_dominated, :], self.ideal_point,
                                                   extreme_points=self.extreme_points)

        # find the intercepts for normalization and do backup if gaussian elimination fails
        worst_of_population = np.max(objs, axis=0)
        worst_of_front = np.max(objs[non_dominated, :], axis=0)

        self.nadir_point = get_nadir_point(self.extreme_points, self.ideal_point, self.worst_point,
                                           worst_of_population, worst_of_front)

        print(f"ideal point is {self.ideal_point}, nadir is {self.nadir_point}, worst is {self.worst_point}")

        #  consider only the population until we come to the splitting front
        I = np.concatenate(fronts)
        print("concat result: ")
        print(I)
        indices_selected[I] = True
        print("indices selected after concat")
        print(indices_selected)
        # update the front indices for the current population
        new_idx_to_old_idx = {}
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                new_idx_to_old_idx[counter] = fronts[i][j]
                fronts[i][j] = counter
                counter += 1
        last_front = fronts[-1]

        # # associate individuals to niches
        # niche_of_individuals, dist_to_niche, dist_matrix = \
        #     associate_to_niches(objs[indices_selected], self.ref_dirs, self.ideal_point, self.nadir_point)
        
        # if we need to select individuals to survive
        #TODO: Need to deal with the case where I is empty (or not a hull)
        if len(objs[indices_selected]) > n_survive:
            objs_to_examine = np.copy(objs[indices_selected])
            #Shift negative to 0
            objs_to_examine = objs_to_examine + objs_to_examine.min(axis=0)
            #Normalize to 1
            objs_to_examine = objs_to_examine / objs_to_examine.max(axis=0)
            if len(fronts) == 1:
                until_last_front = np.argmin(objs_to_examine, axis=0) #Assuming minimization objective
            else:
                until_last_front = np.concatenate(fronts[:-1])
            until_last_front = np.unique(until_last_front)
            n_remaining = n_survive - len(until_last_front)
            until_last_front_list = until_last_front.tolist()
            print("n remaining is {}".format(n_remaining))
            dists = []
            for j in range(len(objs_to_examine)):
                if j in until_last_front_list:
                    continue
                d_obj = []
                for i in range(len(until_last_front)):
                    d_obj.append(np.dot(until_last_front[i], objs_to_examine[j]))
                dists.append(np.min(d_obj))
            print("dists is {}".format(dists))
            dists = np.array(dists)
            survivors = np.concatenate((until_last_front, dists.argsort()[:n_remaining]))
            print("Survivors are {}".format(survivors))
            # Get euclidean distances from each point in objs_to_examine to the points already in the set
            # add n_remaining "closest" points 


            # only survivors need to remain active
            indices_selected[:] = False
            indices_selected[[new_idx_to_old_idx[s] for s in survivors]] = True

        return indices_selected

if __name__ == '__main__':
    gss = MHASubsetSelector(3, n_objs=2)
    objs_cur = np.array([[0, 0], [0.2, 1.5], [0.9, 0],
                         [1.1, 0.2], [1.34, .001], [.001, 1.34],
                         [1, 2], [2, 4], [3, 3],
                         [-1, -1], [-2, -1], [1, -3],
                         [4, 3], [-2, 2], [1, -2],
                         [-0.5, -0.5], [0.5, 0.7], [0.7, 0.5]])
    # objs_cur = np.random.rand(500, 4)
    plt.scatter(objs_cur[:, 0], objs_cur[:, 1])
    idx = gss.select(None, objs_cur)
    print("Indices selected:")
    print(idx)
    print("Vertices selected (by objectives): ")
    print(objs_cur[idx])
    plt.scatter(objs_cur[idx, 0], objs_cur[idx, 1])
    plt.show()