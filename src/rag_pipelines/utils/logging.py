import logging
import os


class LoggerFactory:
    """Factory class to set up and configure a logger with customizable log level and format.

    This class ensures that logging is configured only once during the application's lifetime.
    It can configure the logger based on a given name, logging level, and logging format.

    Attributes:
        logger_name (str): The name of the logger to create.
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        log_format (str): The format for log messages.
        logger (logging.Logger): The configured logger instance.

    Methods:
        get_logger():
            Returns the configured logger instance.
        configure_from_env(logger_name, env_var="LOG_LEVEL"):
            Configures the logger based on an environment variable for dynamic log level setting.
    """

    _is_logger_initialized: bool = False

    def __init__(
        self,
        logger_name: str,
        log_level: int = logging.INFO,
        log_format: str = "%(asctime)s - %(levelname)s - %(message)s",
    ) -> None:
        """Initialize the LoggerFactory instance with the given logger name, log level, and format.

        Args:
            logger_name (str): The name of the logger to create.
            log_level (int, optional): The logging level to use (default is logging.INFO).
            log_format (str, optional): The format for log messages (default is
                                        "%(asctime)s - %(levelname)s - %(message)s").
        """
        self.logger_name = logger_name
        self.log_level = log_level
        self.log_format = log_format
        self.logger = self._initialize_logger()

    def _initialize_logger(self) -> logging.Logger:
        """Configure the logger with the given name and log level.

        Returns:
            logging.Logger: A configured logger instance.
        """
        if not self._is_logger_initialized:
            logging.basicConfig(level=self.log_level, format=self.log_format)
            self._is_logger_initialized = True

        return logging.getLogger(self.logger_name)

    def get_logger(self) -> logging.Logger:
        """Return the configured logger instance.

        Returns:
            logging.Logger: The logger instance.
        """
        return self.logger

    @staticmethod
    def configure_from_env(
        logger_name: str, env_var: str = "LOG_LEVEL"
    ) -> "LoggerFactory":
        """Configure the logger based on an environment variable for dynamic log level setting.

        Args:
            logger_name (str): The name of the logger to create.
            env_var (str, optional): The environment variable to retrieve the log level from (default is "LOG_LEVEL").

        Returns:
            LoggerFactory: A LoggerFactory instance with the log level set based on the environment variable.
        """
        log_level_str = os.getenv(env_var, "INFO").upper()
        # Default to INFO if invalid level
        log_level = getattr(logging, log_level_str, logging.INFO)
        return LoggerFactory(logger_name, log_level=log_level)
