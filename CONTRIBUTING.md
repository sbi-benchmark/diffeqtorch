# Contributing

## Issues

If you encounter problems, open an issue describing the problem. Try to provide sufficient detail for reproducing the issue.


## Pull Requests

For bug fixes and new contributions, open pull requests. If a new feature is contributed, it should generally be covered by tests. Development dependencies can be installed through `pip install "diffeqtorch[dev]"`.


## Tests

To run tests with a custom system image, you can use:

```commandline
$ pytest tests/ --julia-compiled-modules=no --julia-sysimage=$HOME/.julia_sysimage_diffeqtorch.so
```


## Code Formatting

Code should adhere to the [Google Style Guide](http://google.github.io/styleguide/pyguide.html). It should pass through the following tools, which are installed along with `diffeqtorch`:

- **[black](https://github.com/psf/black)**: Automatic code formatting for Python.
- **[isort](https://github.com/timothycrosley/isort)**: Used to consistently order imports.
