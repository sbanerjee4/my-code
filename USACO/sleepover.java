import java.util.*;
import java.io.*;
import java.lang.*;

public class sleepover
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("sleepover.in"));
      int n = in.nextInt();
      boolean[] mark = new boolean[n + 1];
      int i = 1;
      mark[i] = true;
      while(true)
      {
         int dest = (2 * i) % n;
         if(dest == 0) dest = n;
         if(mark[dest]) break;
         mark[dest] = true;
         i = dest;
      }  
      
      System.out.println(i);

      in.close();
   }
}