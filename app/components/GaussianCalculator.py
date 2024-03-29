from statistics import NormalDist, mean, stdev

class GaussianCalculator:

  def __init__(self, data: list) -> None:
    self._mean = self.mean(data)
    self._stdev = self.stdev(data, xbar = self._mean)
    print(f"Mean: {self._mean}\nStdev: {self._stdev}")
    self._pdf = 0

  def calculate_pdf(self, datapoint: int) -> None:
    '''
    Datapoint here refers to the average power of last cycle.
    '''
    self._pdf = NormalDist(self._mean, self._stdev).pdf(datapoint)

  def update(self, data: list) -> None:
    self._mean = self.mean(data)
    self._stdev = self.stdev(data, xbar = self._mean)
    print(f"Mean: {self._mean}\nStdev: {self._stdev}")

  def mean(self, data: list) -> None:
    return mean(data)

  def stdev(self, data: list, xbar: int = None) -> None:
    return stdev(data, xbar = xbar)

  def sigma_rule(self, datapoint: int) -> bool:
    if datapoint > (self._mean + (3*self._stdev)) or datapoint < (self._mean - (3*self._stdev)):
      return True
    else:
      return False