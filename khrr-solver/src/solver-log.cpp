#include "solver-log.h"
#include "solver-fs.h"
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/async.h>
#include <fmt/format.h>
#include <memory>
#include <iostream>

namespace khrr_solver::log {

    void setup_logging() {
        try {
            // Асинхронное логирование для производительности
            spdlog::init_thread_pool(8192, 1);

            // Консольный sink
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_level(spdlog::level::info);
            console_sink->set_pattern("[%H:%M:%S] [%^%l%$] %v");

            // Файловый sink с ротацией (макс 5 файлов по 10 MB)
            auto log_dir = khrr_solver::fs::get_log_directory();
            khrr_solver::fs::create_directory_if_not_exists(log_dir);
            auto log_file = log_dir / fmt::format("{}.log", khrr_solver::fs::APP_NAME);
            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                log_file.string(),
                1024 * 1024 * 10,  // 10 MB
                5                  // 5 файлов
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