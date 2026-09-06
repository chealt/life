use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::thread;

const ADDRESS: &str = "127.0.0.1:8080";
const PUBLIC_DIR: &str = "../public";

fn main() {
    let listener = TcpListener::bind(ADDRESS).expect("Failed to bind to address");
    println!("Server running at http://{}", ADDRESS);
    println!("Serving files from: {}", PUBLIC_DIR);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                thread::spawn(|| {
                    if let Err(e) = handle_connection(stream) {
                        eprintln!("Connection error: {}", e);
                    }
                });
            }
            Err(e) => eprintln!("Failed to establish connection: {}", e),
        }
    }
}

fn handle_connection(mut stream: TcpStream) -> std::io::Result<()> {
    let mut reader = BufReader::new(&stream);
    let mut request_line = String::new();

    // Read the first line of the HTTP request: e.g. "GET /index.html HTTP/1.1"
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }

    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 || parts[0] != "GET" {
        send_response(
            &mut stream,
            405,
            "Method Not Allowed",
            "text/plain",
            b"Method Not Allowed",
        )?;
        return Ok(());
    }

    let raw_path = parts[1].split('?').next().unwrap_or("/");

    // Map root to index.html
    let req_path = if raw_path == "/" {
        "/index.html"
    } else {
        raw_path
    };

    // Sanitize path to prevent directory traversal attacks (e.g. "../../etc/passwd")
    let safe_relative_path = sanitize_path(req_path);
    let file_path = Path::new(PUBLIC_DIR).join(safe_relative_path);

    if file_path.is_file() {
        match fs::read(&file_path) {
            Ok(contents) => {
                let mime_type = get_mime_type(&file_path);
                send_response(&mut stream, 200, "OK", mime_type, &contents)?;
            }
            Err(_) => {
                send_response(
                    &mut stream,
                    500,
                    "Internal Server Error",
                    "text/plain",
                    b"500 Internal Server Error",
                )?;
            }
        }
    } else {
        let not_found_html = b"<!DOCTYPE html><html><body><h1>404 Not Found</h1></body></html>";

        send_response(
            &mut stream,
            404,
            "Not Found",
            "text/html; charset=utf-8",
            not_found_html,
        )?;
    }

    Ok(())
}

fn sanitize_path(path_str: &str) -> PathBuf {
    let mut sanitized = PathBuf::new();
    let path = Path::new(path_str);

    for component in path.components() {
        match component {
            Component::Normal(c) => sanitized.push(c),
            _ => {} // Ignore RootDir, CurDir, and ParentDir ("..")
        }
    }
    sanitized
}

fn get_mime_type(path: &Path) -> &'static str {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        _ => "application/octet-stream",
    }
}

fn send_response(
    stream: &mut TcpStream,
    status_code: u16,
    status_text: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let headers = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status_code,
        status_text,
        content_type,
        body.len()
    );

    stream.write_all(headers.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}
