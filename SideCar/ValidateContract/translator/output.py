
import pandas


def Imp1(p):

    import pandas

    data = pandas.read_csv(p)

    return data


def Imp2(p, data):

    import pandas

    data = data[p].isna().any()

    return data

    @Imp2
    Imp1
