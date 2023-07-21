import abc
import pathlib

class _BaseSaver(abc.ABC):
    """
    Base class to save figures or data
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.out_dir = kwargs.get('out_dir')
        self.out_dir = pathlib.Path(self.out_dir)

        self.modes = kwargs.get('modes', ['train', 'val', 'test'])

        # Make sub directories
        sub_out_dirs = [self.out_dir / mode for mode in self.modes]
        for sub_out_dir in sub_out_dirs:
            if not sub_out_dir.exists():
                sub_out_dir.mkdir(parents=True)

    @abc.abstractmethod
    def save(self, *args, **kwargs):
        raise NotImplementedError()
