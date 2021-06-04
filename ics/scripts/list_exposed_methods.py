import inspect
import pkgutil

import ics


def main():
    package = ics
    for importer, module_name, _ in pkgutil.walk_packages(path=package.__path__,
                                                          prefix=package.__name__ + '.',
                                                          onerror=lambda x: None):
        module = pkgutil.get_loader(module_name).load_module(module_name)
        for class_name, class_ in inspect.getmembers(module, inspect.isclass):
            method_list = inspect.getmembers(class_, lambda x: hasattr(x, 'exposed'))
            if len(method_list) > 0:
                print(f'## {module_name}.{class_name}')
                for method_name, method in method_list:
                    print(f'{class_name}.{method_name}{inspect.signature(method)} {method.__doc__}')


if __name__ == '__main__':
    main()
