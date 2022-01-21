import collections
import re


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def default_dict_depth_hlp(depth, last_factory=lambda: 0):
    if depth == 0:
        return last_factory
    return lambda: collections.defaultdict(default_dict_depth_hlp(depth - 1, last_factory))

    # iterative version does not work - infinite depth
    # lambdas = [last_factory]
    # for i in range(depth):
    #     lambdas.append(lambda: collections.defaultdict(lambdas[i-1]))
    # return lambdas[-1]()


def default_dict_depth(depth, last_factory=lambda: 0):
    return default_dict_depth_hlp(depth, last_factory)()


def iterate_flatmap(dct, depth, _cdepth=0):
    if depth > _cdepth + 1:
        for k in dct.keys():
            r = iterate_flatmap(dct[k], depth, _cdepth=_cdepth + 1)
            for x in r:
                yield (k, *list(x))
    else:
        for k in dct.keys():
            yield k, dct[k]


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def sidak_inv(alpha, m):
    """
    Inverse transformation of sidak_alpha function.
    Used to compute final p-value of M independent tests if while preserving the
    same significance level for the resulting p-value.
    """
    return 1 - (1 - alpha)**m


def merge_pvals(pvals, batch=2):
    """
    Merging pvals with Sidak.

    Note that the merging tree has to be symmetric, otherwise the computation on pvalues is not correct.
    Note: 1-(1-(1-(1-x)^3))^2 == 1-((1-x)^3)^2 == 1-(1-x)^6.
    Example: 12 nodes, binary tree: [12] -> [2,2,2,2,2,2] -> [2,2,2]. So far it is symmetric.
    The next layer of merge is problematic as we merge [2,2] and [2] to two p-values.
    If a minimum is from [2,2] (L) it is a different expression as from [2] R as the lists
    have different lengths. P-value from [2] would increase in significance level compared to Ls on this new layer
    and this it has to be corrected.
    On the other hand, the L minimum has to be corrected as well as it came from
    list of the length 3. We want to obtain 2 p-values which can be merged as if they were equal (exponent 2).
    Thus the exponent on the [2,2] and [2] layer will be 3/2 as we had 3 p-values in total and we are producing 2.
    """
    if len(pvals) <= 1:
        return pvals

    batch = min(max(2, batch), len(pvals))  # norm batch size
    parts = list(chunks(pvals, batch))
    exponent = len(pvals) / len(parts)
    npvals = []
    for p in parts:
        pi = sidak_inv(min(p), exponent)
        npvals.append(pi)
    return merge_pvals(npvals, batch)
