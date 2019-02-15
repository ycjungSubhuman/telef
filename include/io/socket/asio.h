#pragma once

#include <boost/asio.hpp>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
namespace telef::io::socket {
    template<typename SyncReadStream>
    class AsioInputStream : public google::protobuf::io::CopyingInputStream {
    public:
        AsioInputStream(SyncReadStream &sock);

        int Read(void *buffer, int size);

    private:
        SyncReadStream &m_Socket; // Where m_Socket is a instance of boost::asio::ip::tcp::socket
    };


    template<typename SyncReadStream>
    AsioInputStream<SyncReadStream>::AsioInputStream(SyncReadStream &sock) :
            m_Socket(sock) {}


    template<typename SyncReadStream>
    int
    AsioInputStream<SyncReadStream>::Read(void *buffer, int size) {
        std::size_t bytes_read;
        boost::system::error_code ec;
        bytes_read = m_Socket.read_some(boost::asio::buffer(buffer, size), ec);

        if (!ec) {
            return bytes_read;
        } else if (ec == boost::asio::error::eof) {
            return 0;
        } else {
            return -1;
        }
    }


    template<typename SyncWriteStream>
    class AsioOutputStream : public google::protobuf::io::CopyingOutputStream {
    public:
        AsioOutputStream(SyncWriteStream &sock);

        bool Write(const void *buffer, int size);

    private:
        SyncWriteStream &m_Socket; // Where m_Socket is a instance of boost::asio::ip::tcp::socket
    };


    template<typename SyncWriteStream>
    AsioOutputStream<SyncWriteStream>::AsioOutputStream(SyncWriteStream &sock) :
            m_Socket(sock) {}


    template<typename SyncWriteStream>
    bool
    AsioOutputStream<SyncWriteStream>::Write(const void *buffer, int size) {
        boost::system::error_code ec;
        m_Socket.write_some(boost::asio::buffer(buffer, size), ec);
        return !ec;
    }
}