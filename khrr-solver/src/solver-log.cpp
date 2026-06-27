#include "solver-log.h"
#include "solver-fs.h"
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/async.h>
#include <fmt/format.h>
#include <memory>
#include <iostream>

namespace khrr_solver::log {

    std::string get_time_string(void) {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_now;
#ifdef _WIN32
        localtime_s(&tm_now, &time_t_now);
#else
        localtime_r(&time_t_now, &tm_now);
#endif

        char buffer[80];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", &tm_now);

        return buffer;
    }

    void setup_logging() {
        try {
            // Асинхронное логирование для производительности
            spdlog::init_thread_pool(8192, 1);

            // Консольный sink
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_level(spdlog::level::info);
            console_sink->set_pattern("[%H:%M:%S] [%^%l%$] %v");

            // Файловый sink с уникальным именем для каждого запуска
            auto log_dir = khrr_solver::fs::get_log_directory();
            khrr_solver::fs::create_directory_if_not_exists(log_dir);

            // Генерируем уникальное имя файла с временной меткой
            std::string filename = fmt::format(
                "{}_{}.log",
                khrr_solver::fs::APP_NAME,
                get_time_string()
            );
            auto log_file = log_dir / filename;

            auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                log_file.string()
            );
            file_sink->set_level(spdlog::level::trace);
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%L] %v");

            // Объединяем в асинхронный логгер
            std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
            auto logger = std::make_shared<spdlog::async_logger>(
                "multi_sink",
                sinks.begin(),
                sinks.end(),
                spdlog::thread_pool(),
                spdlog::async_overflow_policy::block
            );

            logger->set_level(spdlog::level::trace);
            spdlog::set_default_logger(logger);

            std::cout << fmt::format("Ready to log: {}", log_file.string()) << std::endl;
        }
        catch (const spdlog::spdlog_ex& ex) {
            std::cerr << "Logging setup error : " << ex.what() << std::endl;
        }
    }

}