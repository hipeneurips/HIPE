
COLORS = {
    "fb_sobol": "dodgerblue",
    "fb_random": "purple",
    "nipv": "forestgreen",
    "hipe": "deeppink",
    "lhsbeta": "darkgoldenrod",
    "bald": "darkgoldenrod",
}

NAMES = {
    "hipe": "HIPE",
    "sobol": "Sobol",
    "random": "Random",
    "lhsbeta": "LHS-Beta",
    "nipv": "NIPV",
    "bald": "BALD",
}

BENCHMARKS = {
    "hartmann6": "Hartmann $6D$",
    "hartmann6_12": "Hartmann $6D_e$ ($12D$)",
    "hartmann4": "Hartmann $4D$",
    "hartmann4_8": "Hartmann $4D_e$ ($8D$)",
    "ackley4": "Ackley $4D_e$",
    "Fashion-MNIST": "Fashion-MNIST",
    "higgs": "Higgs",
    "segment": "Segment",
    "MiniBooNE": "MiniBooNE",
    "car": "Car",
    "Australian": "Australian",
    "ishigami": "Ishigami",
    "svm_20": "SVM $20D$",
    "svm_40": "SVM $40D$",
}

METRIC_NAMES = {
    "MLL": "Negative Log-Likelihood",
    "RMSE": "Root Mean Squared Error",
}

BO_METHOD_ORDER = (
    "random",
    "sobol",
    "lhsbeta",
    "nipv",
    "hipe",
)
AL_METHOD_ORDER = (
    "random",
    "sobol",
    "nipv",
    "bald",
    "hipe",
)