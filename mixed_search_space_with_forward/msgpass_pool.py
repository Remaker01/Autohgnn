from search_space.msgpass_pool import HyperMessagePassing, HyperMessagePassingPool as _BaseHMSGPPool


class HyperMessagePassingPool(_BaseHMSGPPool):
    """
    超图消息传递pool
    """
    def get_candidate(self, descrip: str):
        return self.get_msg_passing(descrip)


HMSGPPool = HyperMessagePassingPool
